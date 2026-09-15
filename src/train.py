"""Train SKAPP using validation-only checkpoint selection."""
import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch

from data import Data, metrics, predict
from datasets import NAMES, ROOT, dataset_path, digest, read_manifest
from model import SKAPP


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def train_seed(data, output, seed, args):
    folder = output / f'seed{seed}'
    folder.mkdir()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = SKAPP(prediction_mean_views=data.prediction_mean_views).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    truth = data.splits['valid']['y'].cpu().numpy()
    best, best_epoch = float('inf'), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(data.splits['train']['y']), device=args.device)
        total, count = 0., 0
        for start in range(0, len(order), 128):
            inputs, y = data.batch('train', order[start:start + 128])
            prediction, query, _ = model(*inputs, return_details=True)
            target = (y - data.mean) / data.std
            loss = (prediction.flatten() - target).square().mean() + .1 * (query.flatten() - target).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
        prediction = predict(model, data, 'valid')
        result = metrics(truth, prediction)
        if result['MSE'] < best:
            best, best_epoch = result['MSE'], epoch
            torch.save(model.state_dict(), folder / 'best_model.pth')
            np.save(folder / 'valid_predictions.npy', prediction)
        row = {'epoch': epoch, 'train_objective': total / count, 'valid': result, 'best_epoch': best_epoch}
        history.append(row)
        write_json(folder / 'epochs.json', history)
        print(f'seed{seed}', json.dumps(row), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    summary = {'name': folder.name, 'seed': seed, 'best_epoch': best_epoch, 'epochs': epoch,
               'hidden_dim': 64, 'dropout': .1, 'lr': 3e-4, 'weight_decay': 1e-3,
               'prediction_mean_views': data.prediction_mean_views,
               'retrieval_mean_views': data.retrieval_mean_views,
               'valid': metrics(truth, np.load(folder / 'valid_predictions.npy'))}
    write_json(folder / 'summary.json', summary)
    del model, optimizer
    gc.collect()
    if str(args.device).startswith('cuda'):
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', choices=NAMES)
    source.add_argument('--dataset-path', help='Use a separately prepared dataset directory')
    parser.add_argument('--data-root', help='Download directory; default: datasets/')
    parser.add_argument('--output-dir', help='Default: runs/<dataset>/models')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', nargs='+', type=int, default=[12])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--prediction-mean-views', action=argparse.BooleanOptionalAction, default=None,
                        help='Direct query input; enabled by default for ICIP only.')
    parser.add_argument('--retrieval-mean-views', action=argparse.BooleanOptionalAction, default=None,
                        help='Required prepared retrieval setting; enabled by default for ICIP. Disabling requires rebuilt data.')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use positive epochs/patience and distinct seeds.')
    path = Path(args.dataset_path) if args.dataset_path else dataset_path(args.dataset, args.data_root)
    manifest = read_manifest(path)
    if args.dataset and manifest['dataset'] != args.dataset:
        raise ValueError('Dataset name mismatch')
    output = Path(args.output_dir) if args.output_dir else ROOT / 'runs' / manifest['dataset'] / 'models'
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    is_icip = manifest['dataset'] == 'icip'
    prediction_mv = is_icip if args.prediction_mean_views is None else args.prediction_mean_views
    retrieval_mv = is_icip if args.retrieval_mean_views is None else args.retrieval_mean_views
    data = Data(path, args.device, prediction_mean_views=prediction_mv, retrieval_mean_views=retrieval_mv)
    write_json(output / 'data-manifest.json', {**data.statistics,
               'dataset_manifest_sha256': digest(path / 'dataset.json')})
    state = {'dataset': manifest['dataset'], 'dataset_release': manifest['release'], 'runs': []}
    for seed in args.seeds:
        state['runs'].append(train_seed(data, output, seed, args))
        write_json(output / 'progress.json', state)
    write_json(output / 'summary.json', state)
    print(f'Models saved to {output}')


if __name__ == '__main__':
    main()
