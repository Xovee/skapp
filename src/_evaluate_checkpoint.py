"""Evaluate one validation-selected checkpoint in an isolated process."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data import Data, metrics, predict
from model import SKAPP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', choices=['valid', 'test'], required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = Path(args.manifest)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest['selection_split'] != 'valid':
        raise ValueError('Expected a validation-selected checkpoint')
    checkpoint = path.parent / manifest['checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    expected = json.loads((path.parent / manifest['expected_split_ids']).read_text())
    data = Data(manifest['dataset_path'], args.device, (args.split,), manifest['statistics'],
                prediction_mean_views=manifest['config'].get('prediction_mean_views', False),
                retrieval_mean_views=manifest.get('retrieval_mean_views'))
    ids = data.splits[args.split]['ids']
    if ids != list(map(str, expected[args.split])):
        raise ValueError('Query IDs differ from the fixed split')
    model = SKAPP(**manifest['config']).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    prediction = predict(model, data, args.split)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions')
    truth = data.splits[args.split]['y'].cpu().numpy()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f'{args.split}-predictions.npz',
             image_id=np.asarray(ids), prediction=prediction, target=truth)
    result = {'split': args.split, 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
              'metrics': metrics(truth, prediction), 'fixed_query_ids_verified': True}
    (out / f'{args.split}-metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
