import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_and_save_pkl(input_path, train_path, valid_path, test_path):
    # split the dataset into train/validation/test sets and save them as pickle files
    dataset = pd.read_pickle(input_path)

    train_data, valid_data = train_test_split(dataset, test_size=0.2, random_state=42)
    valid_data, test_data = train_test_split(valid_data, test_size=0.5, random_state=42)

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    train_data.to_pickle(train_path)
    valid_data.to_pickle(valid_path)
    test_data.to_pickle(test_path)


def create_retrieval_pool(train_path, valid_path, retrieval_pool_path):
    # merge train and validation data to form a retrieval pool
    train_data = pd.read_pickle(train_path)
    valid_data = pd.read_pickle(valid_path)

    retrieval_pool = pd.concat([train_data, valid_data], axis=0)
    retrieval_pool.reset_index(drop=True, inplace=True)
    retrieval_pool.to_pickle(retrieval_pool_path)

    return retrieval_pool


def calculate_similarity(query_features, dataset_features, N, list_columns):
    # compute similarity between query and dataset samples based on matching fields
    result = np.zeros((len(dataset_features), len(query_features)), dtype=int)

    for i, feature in enumerate(query_features):
        if i in list_columns:
            # for list-type features, check the intersections
            result[:, i] = [bool(set(feature) & set(df_feature)) for df_feature in dataset_features[:, i]]
        else:
            # for other features, check for equality
            result[:, i] = (dataset_features[:, i] == feature)

    # calculate the number of matches
    n_values = result.sum(axis=0)

    def f_similarity(n):
        return abs( np.log((N - n + 0.5) / (n + 0.5)))

    similarity = np.dot(result, f_similarity(n_values))

    return similarity


def retrieval_data(retrieval_num, data_path, retrieval_pool_path):
    # retrieval top-N similar UGCs for each query UGC in the pool
    dataset = pd.read_pickle(retrieval_pool_path)
    data = pd.read_pickle(data_path)

    all_features = ['user_id', 'date_posted', 'date_taken', 'date_crawl', 'tags', 'contacts',
                    'photo_count', 'mean_views', 'nouns', 'verbs']
    list_columns = [all_features.index(col) for col in ['tags', 'nouns', 'verbs']]

    dataset_array = dataset[all_features].values
    data_array = data[all_features].values
    N = len(dataset)

    retrieved_item_id_list = []
    retrieved_item_similarity_list = []
    retrieved_label_list = []

    for i in tqdm(range(len(data))):
        query_features = data_array[i]
        similarities = calculate_similarity(query_features, dataset_array, N, list_columns)
        similarities[i] = 0  # avoid self-matching
        retrieval_indices = np.argsort(similarities)[::-1][:retrieval_num]
        retrieved_items = dataset.iloc[retrieval_indices]

        retrieved_item_id_list.append(retrieved_items['image_id'].tolist())
        retrieved_item_similarity_list.append(similarities[retrieval_indices].tolist())
        retrieved_label_list.append(retrieved_items['label'].tolist())

    data['retrieved_item_id'] = retrieved_item_id_list
    data['retrieved_item_similarity'] = retrieved_item_similarity_list
    data['retrieved_label'] = retrieved_label_list
    data.to_pickle(data_path)


def _stack_features(df_split, df_database):
    # helper function to stack features
    retrieved_cls_list, retrieved_mean_list, retrieved_text_list, retrieved_label_list = [], [], [], []

    for id_list in tqdm(df_split['retrieved_item_id']):
        cls_list, mean_list, text_list, label_list = [], [], [], []

        for item_id in id_list:
            matched = df_database[df_database['image_id'] == item_id]
            if matched.empty:
                continue
            row = matched.iloc[0]
            cls_list.append(row['cls_vec'])
            mean_list.append(row['mean_pooling_vec'])
            text_list.append(row['merged_text_vec'])
            label_list.append(row['label'])

        retrieved_cls_list.append(cls_list)
        retrieved_mean_list.append(mean_list)
        retrieved_text_list.append(text_list)
        retrieved_label_list.append(label_list)

    df_split['retrieved_visual_feature_embedding_cls'] = retrieved_cls_list
    df_split['retrieved_visual_feature_embedding_mean'] = retrieved_mean_list
    df_split['retrieved_textual_feature_embedding'] = retrieved_text_list
    df_split['retrieved_label_list'] = retrieved_label_list

    return df_split


def stack_retrieved_feature(train_path, valid_path, test_path):
    # stack retrieved features for train, valid, and test sets
    df_train = pd.read_pickle(train_path)
    df_valid = pd.read_pickle(valid_path)
    df_test = pd.read_pickle(test_path)
    df_database = pd.concat([df_train, df_valid, df_test], axis=0).reset_index(drop=True)

    df_train = _stack_features(df_train, df_database)
    df_train.to_pickle(train_path)

    df_valid = _stack_features(df_valid, df_database)
    df_valid.to_pickle(valid_path)

    df_test = _stack_features(df_test, df_database)
    df_test.to_pickle(test_path)


def list2set(path):
    # de-duplicate items in list-type columns
    data = pd.read_pickle(path)
    for col in ['nouns', 'verbs', 'adjectives', 'tags']:
        data[col] = data[col].apply(lambda x: list(set(x)))
    data.to_pickle(path)

    return data


def main():
    start_time = time.time()

    dataset_path = r'datasets/ICIP/dataset.pkl'
    train_path = r'datasets/ICIP/train.pkl'
    valid_path = r'datasets/ICIP/valid.pkl'
    test_path = r'datasets/ICIP/test.pkl'
    retrieval_pool_path = r'datasets/ICIP/retrieval_pool.pkl'

    list2set(dataset_path)
    split_and_save_pkl(dataset_path, train_path, valid_path, test_path)
    print('[1] Split dataset complete.')

    create_retrieval_pool(train_path, valid_path, retrieval_pool_path)
    print('[2] Create retrieval pool complete.')

    retrieval_data(500, train_path, retrieval_pool_path)
    retrieval_data(500, valid_path, retrieval_pool_path)
    retrieval_data(500, test_path, retrieval_pool_path)
    print('[3] Retrieval data complete.')

    stack_retrieved_feature(train_path, valid_path, test_path)
    print('[4] Stack retrieved features complete.')

    # display the runtime
    print(f"Runtime: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
