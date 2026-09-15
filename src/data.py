"""SKAPP data loading with training-only normalization."""
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

from datasets import read_manifest, read_split
from mean_views import retrieval_setting, values_for


def metrics(y, prediction):
    return {'MSE': float(np.mean((y - prediction) ** 2)),
            'MAE': float(np.mean(abs(y - prediction))),
            'SRC': float(spearmanr(y, prediction).statistic) if np.ptp(prediction) > 0 else 0.,
            'samples': len(y)}


class Data:
    def __init__(self, source, device='cpu', splits=('train', 'valid'), statistics=None,
                 neighbors='indices', allow_unready=False, prediction_mean_views=False,
                 retrieval_mean_views=None):
        self.source = Path(source)
        self.device = device
        manifest = read_manifest(source)
        self.prediction_mean_views = prediction_mean_views
        self.retrieval_mean_views = retrieval_setting(source, manifest) if retrieval_mean_views is not None or prediction_mean_views else None
        if prediction_mean_views and manifest['dataset'] != 'icip':
            raise ValueError('mean_views input is only supported for ICIP.')
        if retrieval_mean_views is not None and self.retrieval_mean_views != retrieval_mean_views:
            raise ValueError('Retrieval mean_views setting differs from prepared data. Rebuild neighbors, teachers and RRCP first.')
        if not manifest['rrcp_ready'] and not allow_unready:
            raise ValueError('Compute RRCP with src/build_rrcp.py before training.')
        if neighbors not in ('indices', 'metadata_indices'):
            raise ValueError('Unknown neighbor bank')
        self.statistics = dict(manifest['statistics'])
        self.mean = self.statistics['label_mean']
        self.std = self.statistics['label_std']
        self.rrscale = self.statistics['rrcp_scale_train_only']
        if self.std <= 0 or self.rrscale <= 0:
            raise ValueError('Invalid training normalization')
        bank = read_split(source, 'train', manifest)
        if prediction_mean_views:
            train_values = np.log1p(values_for(bank, 'train'))
            self.statistics.update(mean_views_mean=float(train_values.mean()),
                                   mean_views_std=max(float(train_values.std()), 1e-6))
        if statistics is not None:
            for key, value in self.statistics.items():
                if value != statistics.get(key):
                    raise ValueError(f'Training statistics changed: {key}')
        bank_ids = bank['image_id'].astype(str).tolist()
        positions = {item: index for index, item in enumerate(bank_ids)}
        authors = bank['user_id'].astype(object)
        bank_labels = self.tensor(bank['label'])
        self.bankv = self.tensor(bank['cls_vec'])
        self.bankt = self.tensor(bank['merged_text_vec'])
        self.bankvn = F.normalize(self.bankv, dim=-1)
        self.banktn = F.normalize(self.bankt, dim=-1)
        self.splits = {}
        seen = set()
        for split in splits:
            frame = bank if split == 'train' else read_split(source, split, manifest)
            ids = frame['image_id'].astype(str).tolist()
            if seen.intersection(ids) or (split != 'train' and set(ids).intersection(positions)):
                raise ValueError('Overlapping splits')
            if split == 'train' and ids != bank_ids:
                raise ValueError('Training bank order changed')
            seen.update(ids)
            indices = frame[neighbors]
            if indices.min() < 0 or indices.max() >= len(bank_ids):
                raise ValueError('Neighbor index outside the training bank')
            for start in range(0, len(ids), 2048):
                batch = indices[start:start + 2048]
                self_index = np.asarray([positions.get(item, -1) for item in ids[start:start + 2048]])
                if np.any(batch == self_index[:, None]) or np.any(np.diff(np.sort(batch, axis=1), axis=1) == 0):
                    raise ValueError('Self retrieval or duplicate neighbors')
            qv = self.tensor(frame['mean_pooling_vec'])
            qt = self.tensor(frame['merged_text_vec'])
            qc = self.tensor(frame['cls_vec'])
            ix = self.tensor(indices, torch.long)
            scores = frame['scores']
            scores = (scores - scores.mean(1, keepdims=True)) / np.maximum(scores.std(1, keepdims=True), 1e-3)
            same_author = (authors[indices] == frame['user_id'].astype(object)[:, None]).astype(np.float32)
            relations = torch.empty(len(ids), 500, 4, device=device)
            for start in range(0, len(ids), 128):
                sl = slice(start, start + 128)
                relations[sl, :, 0] = (self.bankvn[ix[sl]] * F.normalize(qc[sl], dim=-1).unsqueeze(1)).sum(-1)
                relations[sl, :, 1] = (self.banktn[ix[sl]] * F.normalize(qt[sl], dim=-1).unsqueeze(1)).sum(-1)
            relations[:, :, 2] = self.tensor(scores.clip(-5, 5))
            relations[:, :, 3] = self.tensor(same_author)
            self.splits[split] = {
                'qv': qv, 'qt': qt, 'ix': ix, 'rel': relations,
                'rr': self.tensor(frame['rrcp'] / self.rrscale),
                'labels': bank_labels[ix], 'y': self.tensor(frame['label']), 'ids': ids,
            }
            if prediction_mean_views:
                value = (np.log1p(values_for(frame, split)) - self.statistics['mean_views_mean']) / self.statistics['mean_views_std']
                self.splits[split]['mean_views'] = self.tensor(value[:, None].astype(np.float32))

    def tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=self.device)

    def batch(self, split, indices, raw=False):
        frame = self.splits[split]
        ix = frame['ix'][indices]
        if raw:
            args = (frame['qv'][indices].unsqueeze(1), frame['qt'][indices].unsqueeze(1),
                    self.bankv[ix].unsqueeze(2), self.bankt[ix].unsqueeze(2),
                    frame['labels'][indices], frame['rr'][indices])
        else:
            args = (F.normalize(frame['qv'][indices], dim=-1), F.normalize(frame['qt'][indices], dim=-1),
                    self.bankvn[ix], self.banktn[ix], (frame['labels'][indices] - self.mean) / self.std,
                    frame['rel'][indices], frame['rr'][indices])
        if not raw and self.prediction_mean_views:
            args = (*args, frame['mean_views'][indices])
        return args, frame['y'][indices]


def predict(model, data, split):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(data.splits[split]['y']), 128):
            args, _ = data.batch(split, slice(start, start + 128))
            prediction = model(*args).flatten() * data.std + data.mean
            values.extend(prediction.cpu().tolist())
    return np.asarray(values, dtype=np.float32)


class TeacherData:
    """Raw features for auxiliary training and RRCP generation."""
    def __init__(self, source, device, splits=('train', 'valid'), include_union=False):
        manifest = read_manifest(source)
        bank = read_split(source, 'train', manifest)
        self.bankv = torch.as_tensor(bank['cls_vec'], device=device)
        self.bankt = torch.as_tensor(bank['merged_text_vec'], device=device)
        self.banky = torch.as_tensor(bank['label'], dtype=torch.float32, device=device)
        positions = {item: index for index, item in enumerate(bank['image_id'].astype(str))}
        self.splits = {}
        seen = set()
        for split in splits:
            frame = bank if split == 'train' else read_split(source, split, manifest)
            ids = frame['image_id'].astype(str).tolist()
            if set(ids).intersection(seen) or (split != 'train' and set(ids).intersection(positions)):
                raise ValueError('Overlapping splits')
            seen.update(ids)
            current = {
                'qv': torch.as_tensor(frame['mean_pooling_vec'], device=device),
                'qt': torch.as_tensor(frame['merged_text_vec'], device=device),
                'y': torch.as_tensor(frame['label'], dtype=torch.float32, device=device), 'ids': ids,
            }
            for key in ['metadata_indices', 'indices'] if include_union else ['metadata_indices']:
                ix = frame[key]
                if ix.min() < 0 or ix.max() >= len(bank['label']):
                    raise ValueError('Invalid neighbor indices')
                for start in range(0, len(ids), 2048):
                    batch = ix[start:start + 2048]
                    own = np.asarray([positions.get(item, -1) for item in ids[start:start + 2048]])
                    if np.any(batch == own[:, None]) or np.any(np.diff(np.sort(batch, axis=1), axis=1) == 0):
                        raise ValueError('Self retrieval or duplicate neighbors')
                current[key] = torch.as_tensor(ix, dtype=torch.long, device=device)
            self.splits[split] = current

    def batch(self, split, query, position=None, neighbors='metadata_indices'):
        frame = self.splits[split]
        ix = frame[neighbors][query] if position is None else frame[neighbors][query, position].unsqueeze(1)
        args = (frame['qv'][query].unsqueeze(1), frame['qt'][query].unsqueeze(1),
                self.bankv[ix].unsqueeze(2), self.bankt[ix].unsqueeze(2), self.banky[ix])
        return args, frame['y'][query]
