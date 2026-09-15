"""Freeze independent attention runs, verify validation parity, then evaluate test.

Each seed is evaluated separately. Aggregate metrics are the arithmetic mean and
sample standard deviation, never metrics computed from averaged predictions.
Use a fresh output directory for each predeclared experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from datasets import NAMES, ROOT, dataset_path as default_dataset_path, read_manifest, split_ids


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def freeze(training_dir, dataset_path, output_dir):
    training_dir = Path(training_dir).resolve()
    output_dir = Path(output_dir).resolve()
    training = json.loads((training_dir / 'summary.json').read_text())
    statistics = json.loads((training_dir / 'data-manifest.json').read_text())
    runs = training['runs']
    if statistics['dataset_manifest_sha256'] != digest(Path(dataset_path) / 'dataset.json'):
        raise ValueError('Dataset changed since training')
    seeds = [run['seed'] for run in runs]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Expected distinct completed training seeds')
    # Check all required inputs before creating the immutable evaluation package.
    for run in runs:
        for name in ['best_model.pth', 'valid_predictions.npy']:
            if not (training_dir / run['name'] / name).is_file():
                raise FileNotFoundError(training_dir / run['name'] / name)
    ids = split_ids(dataset_path)
    if not all(split in ids for split in ['train', 'valid', 'test']):
        raise ValueError('Expected train, valid and test ID lists')
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / 'official-split-ids.json', ids)
    shutil.copyfile(training_dir / 'summary.json', output_dir / 'attention-training.json')
    master = {'selection_split': 'valid', 'prediction_policy': 'independent single attention models',
              'seeds': seeds, 'runs': []}
    for run in runs:
        seed = run['seed']
        folder = training_dir / run['name']
        checkpoint = output_dir / f'attention-seed{seed}.pth'
        shutil.copyfile(folder / 'best_model.pth', checkpoint)
        shutil.copyfile(folder / 'valid_predictions.npy', output_dir / f'validation-reference-seed{seed}.npy')
        manifest = {
            'selection_split': 'valid',
            'dataset_path': str(Path(dataset_path).resolve()),
            'expected_split_ids': 'official-split-ids.json',
            'checkpoint': checkpoint.name, 'sha256': digest(checkpoint),
            'seed': seed, 'best_epoch': run['best_epoch'], 'valid_metrics': run['valid'],
            'config': {key: run[key] for key in ['hidden_dim', 'dropout']},
            'statistics': statistics,
            'retrieval_mean_views': run.get('retrieval_mean_views'),
        }
        manifest['config']['prediction_mean_views'] = run.get('prediction_mean_views', False)
        manifest_path = output_dir / f'manifest-seed{seed}.json'
        write_json(manifest_path, manifest)
        master['runs'].append({'seed': seed, 'manifest': manifest_path.name,
                               'sha256': digest(manifest_path), 'checkpoint': checkpoint.name,
                               'checkpoint_sha256': digest(checkpoint)})
    write_json(output_dir / 'manifest.json', master)
    return master


def summarize(output_dir, seeds):
    results = {}
    for split in ['valid', 'test']:
        runs = {str(seed): json.loads((output_dir / f'evaluation/seed{seed}/{split}-metrics.json').read_text())['metrics']
                for seed in seeds}
        if len({run['samples'] for run in runs.values()}) != 1:
            raise ValueError('Seed sample counts differ')
        results[split] = {
            'runs': runs,
            'metric_mean': {key: float(np.mean([run[key] for run in runs.values()])) for key in ['MSE', 'MAE', 'SRC']},
            'metric_sample_std': {key: float(np.std([run[key] for run in runs.values()], ddof=1))
                                  if len(runs) > 1 else None for key in ['MSE', 'MAE', 'SRC']},
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', choices=NAMES)
    source.add_argument('--dataset-path')
    parser.add_argument('--data-root')
    parser.add_argument('--training-dir', help='Default: runs/<dataset>/models')
    parser.add_argument('--output-dir', help='Default: runs/<dataset>/evaluation')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    path = Path(args.dataset_path) if args.dataset_path else default_dataset_path(args.dataset, args.data_root)
    manifest = read_manifest(path)
    if args.dataset and args.dataset != manifest['dataset']:
        raise ValueError('Dataset name mismatch')
    run_dir = ROOT / 'runs' / manifest['dataset']
    training_dir = Path(args.training_dir) if args.training_dir else run_dir / 'models'
    out = Path(args.output_dir).resolve() if args.output_dir else run_dir / 'evaluation'
    master = freeze(training_dir, path, out)
    state = {'status': 'running', 'manifest_sha256': digest(out / 'manifest.json'), 'completed': []}
    write_json(out / 'progress.json', state)
    evaluator = Path(__file__).with_name('_evaluate_checkpoint.py')
    parity = {}
    try:
        # All validation parity checks must succeed before any test evaluation.
        for split in ['valid', 'test']:
            for run in master['runs']:
                seed = run['seed']
                subprocess.run([sys.executable, str(evaluator), '--manifest', str(out / run['manifest']),
                                '--split', split, '--output-dir', str(out / f'evaluation/seed{seed}'),
                                '--device', args.device], check=True)
                if split == 'valid':
                    predicted = np.load(out / f'evaluation/seed{seed}/valid-predictions.npz')['prediction']
                    np.testing.assert_array_equal(predicted, np.load(out / f'validation-reference-seed{seed}.npy'))
                    parity[str(seed)] = 'bitwise equal'
                state['completed'].append(f'{split}_seed{seed}')
                write_json(out / 'progress.json', state)
            if split == 'valid':
                write_json(out / 'validation-parity.json', parity)
        result = {'status': 'complete', 'results': summarize(out, master['seeds']),
                  'report_policy': 'All seeds, mean and sample standard deviation; no prediction ensemble'}
        write_json(out / 'result.json', result)
        state['status'] = 'complete'
        write_json(out / 'progress.json', state)
        print(json.dumps(result, indent=2))
    except BaseException:
        state['status'] = 'failed'
        write_json(out / 'progress.json', state)
        raise


if __name__ == '__main__':
    main()
