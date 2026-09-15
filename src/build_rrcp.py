"""Generate RRCP with validation-selected auxiliary teachers."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from data import TeacherData
from datasets import NAMES, ROOT, dataset_path, digest, read_manifest, read_split
from teachers import load_teacher, predict_single_item_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', choices=NAMES)
    source.add_argument('--dataset-path')
    parser.add_argument('--data-root')
    parser.add_argument('--all-checkpoint')
    parser.add_argument('--single-checkpoint')
    parser.add_argument('--output-dir', help='Default: runs/<dataset>/data-rebuilt')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    path = Path(args.dataset_path) if args.dataset_path else dataset_path(args.dataset, args.data_root)
    manifest = read_manifest(path)
    if args.dataset and args.dataset != manifest['dataset']:
        raise ValueError('Dataset name mismatch')
    runs = ROOT / 'runs' / manifest['dataset']
    all_path = Path(args.all_checkpoint) if args.all_checkpoint else runs / 'teachers/all/best_model.pth'
    single_path = Path(args.single_checkpoint) if args.single_checkpoint else runs / 'teachers/single/best_model.pth'
    torch.set_num_threads(4)
    all_model = load_teacher(all_path, args.device)
    single_model = load_teacher(single_path, args.device)
    fingerprint = digest(path / 'dataset.json')
    if any(model.dataset_manifest_sha256 != fingerprint for model in [all_model, single_model]):
        raise ValueError('Teachers were trained on a different dataset package')
    if all_model.retrieval_num != 500 or single_model.retrieval_num != 1:
        raise ValueError('Expected all-neighbor and single-neighbor teachers')
    out = Path(args.output_dir) if args.output_dir else runs / 'data-rebuilt'
    out.mkdir(parents=True, exist_ok=False)
    manifest['rrcp_ready'] = False
    (out / 'dataset.json').write_text(json.dumps(manifest, indent=2))
    for split in ['train', 'valid', 'test']:
        data = TeacherData(path, args.device, (split,), include_union=True)
        values = []
        with torch.no_grad():
            for start in range(0, len(data.splits[split]['ids']), 64):
                query = slice(start, start + 64)
                inputs, _ = data.batch(split, query)
                teacher = all_model(*inputs).reshape(-1, 1)
                inputs, _ = data.batch(split, query, neighbors='indices')
                without, with_neighbor = predict_single_item_delta(single_model, *inputs)
                values.extend(((teacher - without).abs() - (teacher - with_neighbor).abs()).cpu().tolist())
        del data
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
        arrays = read_split(path, split)
        arrays['rrcp'] = np.asarray(values, dtype=np.float32)
        if not np.isfinite(arrays['rrcp']).all():
            raise ValueError('Non-finite RRCP')
        if split == 'train':
            manifest['statistics']['rrcp_scale_train_only'] = max(float(np.std(arrays['rrcp'])), 1e-3)
        target = out / f'{split}.npz'
        np.savez(target, **arrays)
        manifest['splits'][split].update(bytes=target.stat().st_size, sha256=digest(target))
        print(f'Generated {split} RRCP', flush=True)
    manifest['rrcp_ready'] = True
    manifest['preparation'].update(query_labels_used_for_rrcp=False,
        all_teacher_sha256=digest(all_path), single_teacher_sha256=digest(single_path))
    (out / 'dataset.json').write_text(json.dumps(manifest, indent=2))
    print(f'Prepared dataset saved to {out}')


if __name__ == '__main__':
    main()
