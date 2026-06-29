import argparse
import heapq
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


SCALAR_FEATURES = [
    "user_id",
    "pathalias",
    "category",
    "subcategory",
    "concepts",
    "postdate",
    "photo_firstdate",
    "photo_firstdatetaken",
    "photo_count",
    "time_zone_id",
]
LIST_FEATURES = ["nouns", "verbs"]


def split_and_save_pkl(input_path, train_path, valid_path, test_path, seed=42):
    dataset = pd.read_pickle(input_path)

    train_data, valid_data = train_test_split(dataset, test_size=0.2, random_state=seed)
    valid_data, test_data = train_test_split(valid_data, test_size=0.5, random_state=seed)

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    train_data.to_pickle(train_path)
    valid_data.to_pickle(valid_path)
    test_data.to_pickle(test_path)


def create_retrieval_pool(train_path, valid_path, retrieval_pool_path):
    retrieval_pool = pd.read_pickle(train_path).copy()
    retrieval_pool.reset_index(drop=True, inplace=True)
    retrieval_pool.to_pickle(retrieval_pool_path)
    return retrieval_pool


def _is_missing(value):
    try:
        return pd.isna(value)
    except ValueError:
        return False


def _scalar_key(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_scalar_key(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_scalar_key(item) for item in value))
    if _is_missing(value):
        return ""
    return value


def _tokens(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [_scalar_key(item) for item in value if not _is_missing(item)]
    if _is_missing(value):
        return []
    return [_scalar_key(value)]


def _bm25_like_weight(pool_size, match_count):
    return math.log((pool_size + 0.5) / (match_count + 0.5))


def _build_indexes(retrieval_pool):
    pool_size = len(retrieval_pool)
    scalar_indexes = {}
    scalar_weights = {}
    list_indexes = {}

    for feature in SCALAR_FEATURES:
        index = defaultdict(list)
        for row_index, value in enumerate(retrieval_pool[feature].tolist()):
            index[_scalar_key(value)].append(row_index)
        scalar_indexes[feature] = dict(index)
        scalar_weights[feature] = {
            key: _bm25_like_weight(pool_size, len(indices))
            for key, indices in index.items()
        }

    for feature in LIST_FEATURES:
        index = defaultdict(list)
        seen_per_token = defaultdict(set)
        for row_index, value in enumerate(retrieval_pool[feature].tolist()):
            for token in _tokens(value):
                if row_index not in seen_per_token[token]:
                    index[token].append(row_index)
                    seen_per_token[token].add(row_index)
        list_indexes[feature] = dict(index)

    return scalar_indexes, scalar_weights, list_indexes


def _score_query(query_row, scalar_indexes, scalar_weights, list_indexes, pool_size):
    scores = defaultdict(float)

    for feature in SCALAR_FEATURES:
        key = _scalar_key(query_row[feature])
        postings = scalar_indexes[feature].get(key, [])
        if not postings:
            continue
        weight = scalar_weights[feature][key]
        for row_index in postings:
            scores[row_index] += weight

    for feature in LIST_FEATURES:
        candidate_indices = set()
        for token in _tokens(query_row[feature]):
            candidate_indices.update(list_indexes[feature].get(token, []))
        if not candidate_indices:
            continue
        weight = _bm25_like_weight(pool_size, len(candidate_indices))
        for row_index in candidate_indices:
            scores[row_index] += weight

    return scores


def _fallback_indices(pool_size, excluded_index, used, needed):
    if needed <= 0:
        return []

    unused_ordered = [
        row_index for row_index in range(pool_size)
        if row_index != excluded_index and row_index not in used
    ]
    all_ordered = [
        row_index for row_index in range(pool_size)
        if row_index != excluded_index
    ]
    if not all_ordered:
        return []

    fallback = []
    for ordered in [unused_ordered, all_ordered]:
        while ordered and len(fallback) < needed:
            take = min(needed - len(fallback), len(ordered))
            fallback.extend(ordered[:take])
    return fallback


def _select_top(scores, retrieval_num, pool_size, excluded_index):
    if excluded_index is not None:
        scores.pop(excluded_index, None)

    scored_items = heapq.nsmallest(
        retrieval_num,
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    selected_indices = [row_index for row_index, _ in scored_items]
    selected_scores = [float(score) for _, score in scored_items]

    if len(selected_indices) < retrieval_num:
        used = set(selected_indices)
        fallback = _fallback_indices(pool_size, excluded_index, used, retrieval_num - len(selected_indices))
        selected_indices.extend(fallback)
        selected_scores.extend([0.0] * len(fallback))

    return selected_indices[:retrieval_num], selected_scores[:retrieval_num]


def retrieval_data(retrieval_num, data_path, retrieval_pool_path):
    retrieval_pool = pd.read_pickle(retrieval_pool_path)
    data = pd.read_pickle(data_path)
    required = set(SCALAR_FEATURES + LIST_FEATURES + ["image_id", "label"])
    missing = required - set(data.columns) | required - set(retrieval_pool.columns)
    if missing:
        raise KeyError(f"SMPD retrieval input is missing required columns: {sorted(missing)}")

    scalar_indexes, scalar_weights, list_indexes = _build_indexes(retrieval_pool)
    pool_size = len(retrieval_pool)
    pool_ids = retrieval_pool["image_id"].astype(str).tolist()
    pool_positions = {image_id: index for index, image_id in enumerate(pool_ids)}
    pool_labels = retrieval_pool["label"].tolist()

    retrieved_item_id_list = []
    retrieved_item_similarity_list = []
    retrieved_label_list = []

    for _, query_row in tqdm(data.iterrows(), total=len(data)):
        query_id = str(query_row["image_id"])
        excluded_index = pool_positions.get(query_id)
        scores = _score_query(query_row, scalar_indexes, scalar_weights, list_indexes, pool_size)
        selected_indices, selected_scores = _select_top(scores, retrieval_num, pool_size, excluded_index)

        retrieved_item_id_list.append([pool_ids[index] for index in selected_indices])
        retrieved_item_similarity_list.append(selected_scores)
        retrieved_label_list.append([pool_labels[index] for index in selected_indices])

    data["retrieved_item_id"] = retrieved_item_id_list
    data["retrieved_item_similarity"] = retrieved_item_similarity_list
    data["retrieved_label"] = retrieved_label_list
    data.to_pickle(data_path)


def stack_retrieved_feature(train_path, valid_path, test_path):
    for split_path in [train_path, valid_path, test_path]:
        df_split = pd.read_pickle(split_path)
        if "retrieved_label_list" not in df_split.columns and "retrieved_label" in df_split.columns:
            df_split["retrieved_label_list"] = df_split["retrieved_label"]
        df_split.to_pickle(split_path)


def list2set(path):
    data = pd.read_pickle(path)
    for column in LIST_FEATURES:
        if column not in data.columns:
            raise KeyError(f"SMPD dataset is missing {column}; run 2_preprocess.py first.")
        data[column] = data[column].apply(lambda value: list(dict.fromkeys(_tokens(value))))
    data.to_pickle(path)
    return data


def run_retrieval(dataset_path, output_dir=None, retrieval_num=50, seed=42):
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir) if output_dir else dataset_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.pkl"
    valid_path = output_dir / "valid.pkl"
    test_path = output_dir / "test.pkl"
    retrieval_pool_path = output_dir / "retrieval_pool.pkl"

    list2set(dataset_path)
    split_and_save_pkl(dataset_path, train_path, valid_path, test_path, seed)
    print("Split dataset done!")

    create_retrieval_pool(train_path, valid_path, retrieval_pool_path)
    print("Create retrieval pool done!")

    retrieval_data(retrieval_num, train_path, retrieval_pool_path)
    retrieval_data(retrieval_num, valid_path, retrieval_pool_path)
    retrieval_data(retrieval_num, test_path, retrieval_pool_path)
    print("Retrieval done!")

    stack_retrieved_feature(train_path, valid_path, test_path)
    print("Stack retrieved feature done!")


def parse_args():
    parser = argparse.ArgumentParser(description="Split SMPD and retrieve top-K train-pool neighbors.")
    parser.add_argument("--dataset_path", default="datasets/SMPD/dataset.pkl", help="Input SMPD dataset pickle")
    parser.add_argument("--output_dir", default=None, help="Output directory for train/valid/test/retrieval_pool")
    parser.add_argument("--retrieval_num", default=50, type=int, help="Number of retrieved UGCs per query")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for train/valid/test split")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    run_retrieval(args.dataset_path, args.output_dir, args.retrieval_num, args.seed)
    print(f"Runtime: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
