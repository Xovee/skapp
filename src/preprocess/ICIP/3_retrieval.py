import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


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

    train_data = pd.read_pickle(train_path)
    valid_data = pd.read_pickle(valid_path)

    retrieval_pool = pd.concat([train_data, valid_data], axis=0)
    retrieval_pool.reset_index(drop=True, inplace=True)

    retrieval_pool.to_pickle(retrieval_pool_path)

    return retrieval_pool


def calculate_similarity(query_features, dataset_features, N, list_columns):
    # 初始化结果数组
    result = np.zeros((len(dataset_features), len(query_features)), dtype=int)

    for i, feature in enumerate(query_features):
        if i in list_columns:
            # 对于列表类型的特征，使用集合判断是否有交集
            result[:, i] = [bool(set(feature) & set(df_feature)) for df_feature in dataset_features[:, i]]
        else:
            # 对于其他类型的特征，直接比较是否相等
            result[:, i] = (dataset_features[:, i] == feature)

    # 计算每个特征值出现的次数
    n_values = result.sum(axis=0)

    def f_similarity(n):
        # 计算相似度
        return abs( np.log((N - n + 0.5) / (n + 0.5)))

    # 计算相似度
    similarity = np.dot(result, f_similarity(n_values))
    return similarity


def retrieval_data(retrieval_num, data_path, retrieval_pool_path):
    # 读取数据集和待检索数据
    dataset = pd.read_pickle(retrieval_pool_path)
    data = pd.read_pickle(data_path)

    # 获取所有特征列
    all_features = ['user_id', 'date_posted', 'date_taken', 'date_crawl', 'tags', 'contacts',
                    'photo_count', 'mean_views', 'nouns', 'verbs']

    # 指定列表类型的列
    list_columns = [all_features.index(col) for col in ['tags', 'nouns', 'verbs']]

    # 转换为 Numpy 数组，以便进行高效的向量化操作
    dataset_array = dataset[all_features].values
    data_array = data[all_features].values

    # 计算数据集大小
    N = len(dataset)

    # 存储检索结果的列表
    retrieved_item_id_list = []
    retrieved_item_similarity_list = []
    retrieved_label_list = []

    # 遍历待检索数据
    for i in tqdm(range(len(data))):
        # 获取查询特征向量
        query_features = data_array[i]

        # 计算相似度
        similarities = calculate_similarity(query_features, dataset_array, N, list_columns)

        # 将自身相似度置为0
        similarities[i] = 0

        # 获取相似度排序后的索引
        retrieval_indices = np.argsort(similarities)[::-1][:retrieval_num]
        retrieved_items = dataset.iloc[retrieval_indices]

        # 提取检索结果的相关信息，并存储到列表中
        retrieved_item_id_list.append(retrieved_items['image_id'].tolist())
        retrieved_item_similarity_list.append(similarities[retrieval_indices].tolist())
        retrieved_label_list.append(retrieved_items['label'].tolist())

    # 将检索结果存储到待检索数据中
    data['retrieved_item_id'] = retrieved_item_id_list
    data['retrieved_item_similarity'] = retrieved_item_similarity_list
    data['retrieved_label'] = retrieved_label_list


    # 存储结果到文件
    data.to_pickle(data_path)


def stack_retrieved_feature(train_path, valid_path, test_path):

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


def list2set(path):

    data = pd.read_pickle(path)
    data['nouns'] = data['nouns'].apply(lambda x: list(set(x)))
    data['verbs'] = data['verbs'].apply(lambda x: list(set(x)))
    data['adjectives'] = data['adjectives'].apply(lambda x: list(set(x)))
    data['tags'] = data['tags'].apply(lambda x: list(set(x)))
    data.to_pickle(path)
    return data


if __name__ == "__main__":

    dataset_path = r'datasets/ICIP/dataset.pkl'
    train_path = r'datasets/ICIP/train.pkl'
    valid_path = r'datasets/ICIP/valid.pkl'
    test_path = r'datasets/ICIP/test.pkl'
    retrieval_pool_path = r'datasets/ICIP/retrieval_pool.pkl'

    list2set(dataset_path)

    split_and_save_pkl(dataset_path, train_path, valid_path, test_path)
    print('Split dataset done!')

    create_retrieval_pool(train_path, valid_path, retrieval_pool_path)
    print('Create retrieval pool done!')

    retrieval_data(500, train_path, retrieval_pool_path)
    retrieval_data(500, valid_path, retrieval_pool_path)
    retrieval_data(500, test_path, retrieval_pool_path)
    print('Retrieval done!')

    stack_retrieved_feature(train_path, valid_path, test_path)
    print('Stack retrieved feature done!')
