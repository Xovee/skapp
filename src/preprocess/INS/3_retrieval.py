import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse



def split_and_save_pkl(input_path, train_path, valid_path, test_path):

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

    retrieval_pool = pd.read_pickle(train_path).copy()
    retrieval_pool.reset_index(drop=True, inplace=True)

    retrieval_pool.to_pickle(retrieval_pool_path)

    return retrieval_pool


def calculate_similarity(all_retrieval_result, N):
    # Count how often each feature value appears in the retrieval pool.
    n_values = all_retrieval_result.sum(axis=0)

    def f_similarity(n):
        # Compute the feature-level similarity weight.
        return np.log((N - n + 0.5) / (n + 0.5))

    # Compute the final similarity score.
    similarity = np.dot(all_retrieval_result, f_similarity(n_values))
    return similarity


def retrieval_data(retrieval_num, data_path, retrieval_pool_path):
    # Load the retrieval pool and query split.
    dataset = pd.read_pickle(retrieval_pool_path)
    data = pd.read_pickle(data_path)

    # Select metadata fields used for retrieval.
    all_features = ['comment_num', 'user_id', 'taken_timestamp']

    # Convert fields to NumPy arrays for vectorized matching.
    dataset_array = dataset[all_features].values
    data_array = data[all_features].values
    retrieval_ids = dataset['image_id'].astype(str).to_numpy()
    data_ids = data['image_id'].astype(str).to_numpy()

    # Record retrieval pool size.
    N = len(dataset)

    # Store retrieved items, scores, and labels.
    retrieved_item_id_list = []
    retrieved_item_similarity_list = []
    retrieved_label_list = []

    # Retrieve neighbors for each query item.
    for i in tqdm(range(len(data))):
        # Get query metadata features.
        query_features = data_array[i]

        # Compute metadata similarity.
        similarities = calculate_similarity((dataset_array == query_features).astype(int), N)
        similarities[retrieval_ids == data_ids[i]] = -np.inf

        # Select top-ranked retrieval indices.
        retrieval_indices = np.argsort(similarities)[::-1][:retrieval_num]
        retrieved_items = dataset.iloc[retrieval_indices]

        # Save retrieved item metadata.
        retrieved_item_id_list.append(retrieved_items['image_id'].tolist())
        retrieved_item_similarity_list.append(similarities[retrieval_indices].tolist())
        retrieved_label_list.append(retrieved_items['label'].tolist())

    # Attach retrieval results to the query split.
    data['retrieved_item_id'] = retrieved_item_id_list
    data['retrieved_item_similarity'] = retrieved_item_similarity_list
    data['retrieved_label'] = retrieved_label_list

    # Save retrieval results.
    data.to_pickle(data_path)

def stack_retrieved_feature(train_path, valid_path, test_path):
    for split_path in [train_path, valid_path, test_path]:
        df_split = pd.read_pickle(split_path)
        if 'retrieved_label_list' not in df_split.columns and 'retrieved_label' in df_split.columns:
            df_split['retrieved_label_list'] = df_split['retrieved_label']
        df_split.to_pickle(split_path)
    return

    df_train = pd.read_pickle(train_path)

    df_test = pd.read_pickle(test_path)

    df_valid = pd.read_pickle(valid_path)

    df_database = pd.concat([df_train, df_test, df_valid], axis=0)

    df_database.reset_index(drop=True, inplace=True)

    retrieved_visual_feature_embedding_cls_list = []

    retrieved_visual_feature_embedding_mean_list = []

    retrieved_textual_feature_embedding_list = []

    retrieve_label_list = []

    for i in tqdm(range(len(df_train))):

        id_list = df_train['retrieved_item_id'][i]

        current_retrieved_visual_feature_embedding_cls_list = []

        current_retrieved_visual_feature_embedding_mean_list = []

        current_retrieved_textual_feature_embedding_list = []

        current_retrieved_label_list = []

        for j in range(len(id_list)):
            item_id = id_list[j]

            index = df_database[df_database['image_id'] == item_id].index[0]

            current_retrieved_visual_feature_embedding_cls_list.append(
                df_database['cls_vec'][index])

            current_retrieved_visual_feature_embedding_mean_list.append(
                df_database['mean_pooling_vec'][index])

            current_retrieved_textual_feature_embedding_list.append(df_database['merged_text_vec'][index])

            current_retrieved_label_list.append(df_database['label'][index])

        retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)

        retrieved_visual_feature_embedding_mean_list.append(current_retrieved_visual_feature_embedding_mean_list)

        retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)

        retrieve_label_list.append(current_retrieved_label_list)

    df_train['retrieved_visual_feature_embedding_cls'] = retrieved_visual_feature_embedding_cls_list

    df_train['retrieved_visual_feature_embedding_mean'] = retrieved_visual_feature_embedding_mean_list

    df_train['retrieved_textual_feature_embedding'] = retrieved_textual_feature_embedding_list

    df_train['retrieved_label_list'] = retrieve_label_list

    df_train.to_pickle(train_path)

    retrieved_visual_feature_embedding_cls_list = []

    retrieved_visual_feature_embedding_mean_list = []

    retrieved_textual_feature_embedding_list = []

    retrieve_label_list = []

    for i in tqdm(range(len(df_test))):

        id_list = df_test['retrieved_item_id'][i]

        current_retrieved_visual_feature_embedding_cls_list = []

        current_retrieved_visual_feature_embedding_mean_list = []

        current_retrieved_textual_feature_embedding_list = []

        current_retrieved_label_list = []

        for j in range(len(id_list)):
            item_id = id_list[j]

            index = df_database[df_database['image_id'] == item_id].index[0]

            current_retrieved_visual_feature_embedding_cls_list.append(
                df_database['cls_vec'][index])

            current_retrieved_visual_feature_embedding_mean_list.append(
                df_database['mean_pooling_vec'][index])

            current_retrieved_textual_feature_embedding_list.append(df_database['merged_text_vec'][index])

            current_retrieved_label_list.append(df_database['label'][index])

        retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)

        retrieved_visual_feature_embedding_mean_list.append(current_retrieved_visual_feature_embedding_mean_list)

        retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)

        retrieve_label_list.append(current_retrieved_label_list)

    df_test['retrieved_visual_feature_embedding_cls'] = retrieved_visual_feature_embedding_cls_list

    df_test['retrieved_visual_feature_embedding_mean'] = retrieved_visual_feature_embedding_mean_list

    df_test['retrieved_textual_feature_embedding'] = retrieved_textual_feature_embedding_list

    df_test['retrieved_label_list'] = retrieve_label_list

    df_test.to_pickle(test_path)

    retrieved_visual_feature_embedding_cls_list = []

    retrieved_visual_feature_embedding_mean_list = []

    retrieved_textual_feature_embedding_list = []

    retrieve_label_list = []

    for i in tqdm(range(len(df_valid))):

        id_list = df_valid['retrieved_item_id'][i]

        current_retrieved_visual_feature_embedding_cls_list = []

        current_retrieved_visual_feature_embedding_mean_list = []

        current_retrieved_textual_feature_embedding_list = []

        current_retrieved_label_list = []

        for j in range(len(id_list)):
            item_id = id_list[j]

            index = df_database[df_database['image_id'] == item_id].index[0]

            current_retrieved_visual_feature_embedding_cls_list.append(
                df_database['cls_vec'][index])

            current_retrieved_visual_feature_embedding_mean_list.append(
                df_database['mean_pooling_vec'][index])

            current_retrieved_textual_feature_embedding_list.append(df_database['merged_text_vec'][index])

            current_retrieved_label_list.append(df_database['label'][index])

        retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)

        retrieved_visual_feature_embedding_mean_list.append(current_retrieved_visual_feature_embedding_mean_list)

        retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)

        retrieve_label_list.append(current_retrieved_label_list)

    df_valid['retrieved_visual_feature_embedding_cls'] = retrieved_visual_feature_embedding_cls_list

    df_valid['retrieved_visual_feature_embedding_mean'] = retrieved_visual_feature_embedding_mean_list

    df_valid['retrieved_textual_feature_embedding'] = retrieved_textual_feature_embedding_list

    df_valid['retrieved_label_list'] = retrieve_label_list

    df_valid.to_pickle(valid_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Split Instagram and retrieve top-K train-pool neighbors.")
    parser.add_argument("--retrieval_num", default=50, type=int, help="Number of retrieved UGCs per query")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_path = r'datasets/INS/dataset.pkl'
    train_path = r'datasets/INS/train.pkl'
    valid_path = r'datasets/INS/valid.pkl'
    test_path = r'datasets/INS/test.pkl'
    retrieval_pool_path = r'datasets/INS/retrieval_pool.pkl'

    split_and_save_pkl(dataset_path, train_path, valid_path, test_path)
    print('Split dataset done!')

    create_retrieval_pool(train_path, valid_path, retrieval_pool_path)
    print('Create retrieval pool done!')

    retrieval_data(args.retrieval_num, train_path, retrieval_pool_path)
    retrieval_data(args.retrieval_num, valid_path, retrieval_pool_path)
    retrieval_data(args.retrieval_num, test_path, retrieval_pool_path)

    print('Retrieval done!')

    stack_retrieved_feature(train_path, valid_path, test_path)

    print('Stack retrieved feature done!')
