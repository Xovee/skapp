"""ICIP metadata and precomputed retrieval protocol validation."""
import numpy as np


def retrieval_setting(source, manifest):
    if manifest['dataset'] != 'icip':
        return False
    value = manifest.get('retrieval_mean_views', manifest.get('preparation', {}).get('ablation_retrieval_mean_views'))
    if type(value) is not bool:
        raise ValueError('ICIP dataset must declare retrieval_mean_views for its precomputed neighbors and RRCP.')
    return value


def values_for(frame, split):
    if 'mean_views' not in frame:
        raise ValueError('ICIP data is missing mean_views; download the current dataset.')
    values = np.asarray(frame['mean_views'], dtype=np.float64)
    if values.shape != (len(frame['image_id']),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('mean_views must be finite, nonnegative, and aligned with sample IDs.')
    return values
