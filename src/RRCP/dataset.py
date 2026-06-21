import torch.utils.data
import numpy as np
import pandas as pd
from pathlib import Path


def custom_collate_fn(batch):
    mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
        retrieved_textual_feature_embedding, retrieved_label_list, label = zip(*batch)

    return torch.from_numpy(np.asarray(mean_pooling_vec, dtype=np.float32)),\
            torch.from_numpy(np.asarray(merge_text_vec, dtype=np.float32)),\
            torch.from_numpy(np.asarray(retrieved_visual_feature_embedding_cls, dtype=np.float32)),\
            torch.from_numpy(np.asarray(retrieved_textual_feature_embedding, dtype=np.float32)),\
            torch.from_numpy(np.asarray(retrieved_label_list, dtype=np.float32)),\
            torch.from_numpy(np.asarray(label, dtype=np.float32)).unsqueeze(-1)


def _stack_feature(series):
    return np.asarray(series.tolist(), dtype=np.float32)


def _build_retrieval_indices(id_lists, retrieval_pool_ids, retrieval_num):
    id_array = np.asarray([item_ids[:retrieval_num] for item_ids in id_lists], dtype=object)
    pool_ids = pd.Index(retrieval_pool_ids)
    pool_positions = pd.Series(np.arange(len(pool_ids)), index=pool_ids)
    pool_positions = pool_positions[~pool_positions.index.duplicated(keep='last')]
    flat_indices = pool_positions.reindex(id_array.reshape(-1)).to_numpy()
    missing_mask = pd.isna(flat_indices)
    if missing_mask.any():
        missing_id = id_array.reshape(-1)[np.where(missing_mask)[0][0]]
        raise KeyError(f'Retrieved item id not found in retrieval_pool.pkl: {missing_id}')
    return flat_indices.reshape(id_array.shape).astype(np.int64, copy=False)


def _retrieved_labels(dataframe):
    label_column = 'retrieved_label_list' if 'retrieved_label_list' in dataframe.columns else 'retrieved_label'
    return np.asarray(dataframe[label_column].tolist(), dtype=np.float32)


class MyData(torch.utils.data.Dataset):

    def __init__(self, retrieval_num, path):
        super().__init__()

        self.path = Path(path)
        self.retrieval_num = int(retrieval_num)
        self.dynamic_single_item = False

        if self._init_dynamic_single_item():
            return

        self.dataframe = pd.read_pickle(self.path)
        self.length = len(self.dataframe)
        self.label = self.dataframe['label'].to_numpy(dtype=np.float32)
        self.mean_pooling_vec = _stack_feature(self.dataframe['mean_pooling_vec'])
        self.merge_text_vec = _stack_feature(self.dataframe['merged_text_vec'])
        self.retrieval_label_list = _retrieved_labels(self.dataframe)
        self.use_feature_bank = self._init_feature_bank(self.dataframe)

        if not self.use_feature_bank:
            self.retrieval_visual_feature_embedding_cls = self.dataframe['retrieved_visual_feature_embedding_cls']
            self.retrieval_textual_feature_embedding = self.dataframe['retrieved_textual_feature_embedding']
        self.dataframe = None

    def _source_path_for_dynamic_single_item(self):
        parent = self.path.parent
        if not parent.name.endswith('_dissembled'):
            return None
        source_dir = parent.with_name(parent.name[:-len('_dissembled')])
        source_path = source_dir / self.path.name
        return source_path if source_path.exists() else None

    def _init_dynamic_single_item(self):
        source_path = self._source_path_for_dynamic_single_item()
        if source_path is None:
            return False

        self.dynamic_single_item = True
        self.dataframe = pd.read_pickle(source_path)
        self.label = self.dataframe['label'].to_numpy(dtype=np.float32)
        self.mean_pooling_vec = _stack_feature(self.dataframe['mean_pooling_vec'])
        self.merge_text_vec = _stack_feature(self.dataframe['merged_text_vec'])
        self.retrieval_label_list = _retrieved_labels(self.dataframe)
        self.source_retrieval_num = self.retrieval_label_list.shape[1]
        self.use_feature_bank = self._init_feature_bank(self.dataframe)

        if not self.use_feature_bank:
            self.retrieval_visual_feature_embedding_cls = self.dataframe['retrieved_visual_feature_embedding_cls']
            self.retrieval_textual_feature_embedding = self.dataframe['retrieved_textual_feature_embedding']

        self.order = np.arange(len(self.dataframe) * self.source_retrieval_num, dtype=np.int64)
        np.random.RandomState(42).shuffle(self.order)
        self.length = len(self.order)
        self.dataframe = None
        return True

    def _init_feature_bank(self, dataframe):
        pool_path = self.path.parent / 'retrieval_pool.pkl'
        if self.dynamic_single_item:
            pool_path = self.path.parent.with_name(self.path.parent.name[:-len('_dissembled')]) / 'retrieval_pool.pkl'
        if 'retrieved_item_id' not in dataframe.columns or not pool_path.exists():
            return False

        retrieval_pool = pd.read_pickle(pool_path)
        required_columns = {'image_id', 'cls_vec', 'merged_text_vec'}
        if not required_columns.issubset(retrieval_pool.columns):
            return False

        retrieval_num = self.source_retrieval_num if self.dynamic_single_item else self.retrieval_num
        self.retrieved_indices = _build_retrieval_indices(
            dataframe['retrieved_item_id'], retrieval_pool['image_id'].tolist(), retrieval_num
        )
        self.feature_bank_visual = _stack_feature(retrieval_pool['cls_vec'])
        self.feature_bank_textual = _stack_feature(retrieval_pool['merged_text_vec'])
        return True

    def __getitem__(self, item):
        if self.dynamic_single_item:
            flat_index = self.order[item]
            row = flat_index // self.source_retrieval_num
            retrieval_idx = flat_index % self.source_retrieval_num
            label = self.label[row]
            mean_pooling_vec = self.mean_pooling_vec[row]
            merge_text_vec = self.merge_text_vec[row]
            if self.use_feature_bank:
                feature_bank_idx = self.retrieved_indices[row, retrieval_idx]
                retrieved_visual_feature_embedding_cls = self.feature_bank_visual[feature_bank_idx]
                retrieved_textual_feature_embedding = self.feature_bank_textual[feature_bank_idx]
            else:
                retrieved_visual_feature_embedding_cls = self.retrieval_visual_feature_embedding_cls[row][retrieval_idx]
                retrieved_textual_feature_embedding = self.retrieval_textual_feature_embedding[row][retrieval_idx]
            retrieved_label_list = [self.retrieval_label_list[row, retrieval_idx]]

            return mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
                   retrieved_textual_feature_embedding, retrieved_label_list, label

        label = self.label[item]
        mean_pooling_vec = self.mean_pooling_vec[item]
        merge_text_vec = self.merge_text_vec[item]
        if self.use_feature_bank:
            retrieved_indices = self.retrieved_indices[item]
            retrieved_visual_feature_embedding_cls = self.feature_bank_visual[retrieved_indices]
            retrieved_textual_feature_embedding = self.feature_bank_textual[retrieved_indices]
        else:
            retrieved_visual_feature_embedding_cls = self.retrieval_visual_feature_embedding_cls[item]
            retrieved_textual_feature_embedding = self.retrieval_textual_feature_embedding[item]
        retrieved_label_list = self.retrieval_label_list[item][:self.retrieval_num]

        return mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
               retrieved_textual_feature_embedding, retrieved_label_list, label

    def __len__(self):
        if self.dynamic_single_item:
            return self.length
        return self.length
