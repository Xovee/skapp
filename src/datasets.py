"""Dataset locations and portable NumPy input files."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('icip', 'smpd', 'instagram')


def dataset_path(name, data_root=None):
    return (Path(data_root) if data_root else ROOT / 'datasets') / name


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_manifest(source):
    path = Path(source) / 'dataset.json'
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found at {source}. Run src/download.py first.')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['format_version'] != 1 or manifest['dataset'] not in NAMES:
        raise ValueError('Unsupported dataset format')
    if manifest['neighbors'] != 500 or manifest['feature_dim'] != 768:
        raise ValueError('Unsupported input dimensions')
    return manifest


def read_split(source, split, manifest=None):
    manifest = read_manifest(source) if manifest is None else manifest
    entry = manifest['splits'][split]
    if entry['file'] != f'{split}.npz':
        raise ValueError('Invalid split filename')
    file = Path(source) / entry['file']
    if file.stat().st_size != entry['bytes'] or digest(file) != entry['sha256']:
        raise ValueError(f'{split} input checksum mismatch; download the dataset again.')
    with np.load(file, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    ids = arrays['image_id'].astype(str).tolist()
    if len(ids) != entry['rows'] or len(set(ids)) != len(ids):
        raise ValueError(f'Invalid {split} sample IDs')
    if hashlib.sha256('\n'.join(ids).encode()).hexdigest() != entry['ids_sha256']:
        raise ValueError(f'{split} sample order changed')
    if arrays['user_id'].shape != (len(ids),) or arrays['label'].shape != (len(ids),):
        raise ValueError('Invalid sample metadata')
    if manifest['dataset'] == 'icip':
        value = arrays.get('mean_views')
        if value is None or value.shape != (len(ids),) or value.dtype.kind not in 'fi' or (value < 0).any():
            raise ValueError('ICIP requires aligned nonnegative mean_views values; download the current dataset.')
    for key in ['cls_vec', 'mean_pooling_vec', 'merged_text_vec']:
        if arrays[key].shape != (len(ids), 768) or arrays[key].dtype != np.float32:
            raise ValueError(f'Invalid {key} features')
    for key in ['indices', 'metadata_indices', 'scores', 'rrcp']:
        if arrays[key].shape != (len(ids), 500):
            raise ValueError(f'Invalid {key} neighbors')
    for key in ['indices', 'metadata_indices']:
        if arrays[key].dtype.kind not in 'iu':
            raise ValueError('Neighbor indices must be integers')
    if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in 'fi'):
        raise ValueError('Non-finite dataset values')
    return arrays


def split_ids(source):
    manifest = read_manifest(source)
    result = {}
    for split, entry in manifest['splits'].items():
        with np.load(Path(source) / entry['file'], allow_pickle=False) as stored:
            result[split] = stored['image_id'].astype(str).tolist()
    return result
