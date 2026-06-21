import torch.utils.data
import numpy as np
import pandas as pd
from pathlib import Path


def custom_collate_fn(batch):
    mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
        retrieved_textual_feature_embedding, retrieved_label_list, RRCP, label = zip(*batch)

    return torch.from_numpy(np.asarray(mean_pooling_vec, dtype=np.float32)), \
        torch.from_numpy(np.asarray(merge_text_vec, dtype=np.float32)), \
        torch.from_numpy(np.asarray(retrieved_visual_feature_embedding_cls, dtype=np.float32)), \
        torch.from_numpy(np.asarray(retrieved_textual_feature_embedding, dtype=np.float32)), \
        torch.from_numpy(np.asarray(retrieved_label_list, dtype=np.float32)), \
        torch.from_numpy(np.asarray(RRCP, dtype=np.float32)), \
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
        self.dataframe = pd.read_pickle(self.path)
        self.length = len(self.dataframe)
        self.label = self.dataframe['label'].to_numpy(dtype=np.float32)
        self.mean_pooling_vec = _stack_feature(self.dataframe['mean_pooling_vec'])
        self.merge_text_vec = _stack_feature(self.dataframe['merged_text_vec'])
        self.retrieval_label_list = _retrieved_labels(self.dataframe)
        self.RRCP = np.asarray(self.dataframe['RRCP_silver'].tolist(), dtype=np.float32)
        self.use_feature_bank = self._init_feature_bank()

        if not self.use_feature_bank:
            self.retrieval_visual_feature_embedding_cls = self.dataframe['retrieved_visual_feature_embedding_cls']
            self.retrieval_textual_feature_embedding = self.dataframe['retrieved_textual_feature_embedding']
        self.dataframe = None

    def _init_feature_bank(self):
        pool_path = self.path.parent / 'retrieval_pool.pkl'
        if 'retrieved_item_id' not in self.dataframe.columns or not pool_path.exists():
            return False

        retrieval_pool = pd.read_pickle(pool_path)
        required_columns = {'image_id', 'cls_vec', 'merged_text_vec'}
        if not required_columns.issubset(retrieval_pool.columns):
            return False

        self.retrieved_indices = _build_retrieval_indices(
            self.dataframe['retrieved_item_id'], retrieval_pool['image_id'].tolist(), self.retrieval_num
        )
        self.feature_bank_visual = _stack_feature(retrieval_pool['cls_vec'])
        self.feature_bank_textual = _stack_feature(retrieval_pool['merged_text_vec'])
        return True

    def __getitem__(self, item):

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
        RRCP = self.RRCP[item][:self.retrieval_num]

        return mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
            retrieved_textual_feature_embedding, retrieved_label_list, RRCP, label

    def __len__(self):
        return self.length
