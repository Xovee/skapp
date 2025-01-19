import pandas as pd
import math
from tqdm import tqdm
import re
import os

def encode_tags(word_list):
    word_dict = {}
    encoded_list = []

    for sublist in word_list:

        if sublist == []:
            encoded_list.append([0])
            continue

        if isinstance(sublist, list):
            words_list = sublist
        else:
            words_list = eval(sublist)

        words = []

        for word in words_list:
            if word not in word_dict:
                word_dict[word] = len(word_dict) + 1

            words.append(word_dict[word])
        encoded_list.append(words)

    return encoded_list


def process_meta_data(headers, img_info, popularity, path):

    merged_data_1 = pd.merge(headers, img_info, on='FlickrId')

    merged_data = pd.merge(merged_data_1, popularity, on='FlickrId')

    merged_data['label'] = [math.log2(merged_data['Day30'][i] / 30 + 1) for i in range(len(merged_data['Day30']))]

    tags = encode_tags(merged_data['Tags'])

    title = merged_data['Title']

    description = merged_data['Description']

    text = [str(x) + ' ' + str(y) for x, y in zip(title, description)]

    pattern = r'<a[^>]*>(.*?)</a>'

    clean_text = [re.sub(pattern, '', item) for item in text]

    avg_group_members = [int(item) for item in merged_data['AvgGroupsMemb']]

    avg_group_photos = [int(item) for item in merged_data['AvgGroupPhotos']]

    dataset = {
        'image_id': merged_data['FlickrId'],
        'text': clean_text,
        'label': merged_data['label'],
        'user_id': merged_data['UserId'],
        'date_posted': merged_data['DatePosted'],
        'date_taken': merged_data['DateTaken'],
        'date_crawl': merged_data['DateCrawl'],
        'size': merged_data['Size'],
        'num_sets': merged_data['NumSets'],
        'num_groups': merged_data['NumGroups'],
        'avg_group_members': avg_group_members,
        'avg_group_photos': avg_group_photos,
        'tags': tags
    }

    df = pd.DataFrame(dataset)

    df.to_pickle(path)

    return df


def process_user_data(dataset, users, path):


    user_id = dataset['user_id']

    is_pro_list = []
    has_status_list = []
    contacts_list = []
    photo_count_list = []
    mean_views_list = []

    for i in tqdm(range(len(user_id))):

        for j in range(len(users['UserId'])):

            if user_id[i] == users['UserId'][j]:
                is_pro_list.append(users['Ispro'][j])
                has_status_list.append(users['HasStats'][j])
                contacts_list.append(users['Contacts'][j])
                photo_count_list.append(users['PhotoCount'][j])
                mean_views_list.append(users['MeanViews'][j])

                break

    mean_views_list = [int(item) for item in mean_views_list]

    dataset['is_pro'] = is_pro_list
    dataset['has_status'] = has_status_list
    dataset['contacts'] = contacts_list
    dataset['photo_count'] = photo_count_list
    dataset['mean_views'] = mean_views_list

    dataset.to_pickle(path)

    return dataset


if __name__ == "__main__":

    path = r"datasets/ICIP/dataset.pkl"
    dataset_path = r"datasets/origin_dataset/ICIP"

    users = pd.read_csv(os.path.join(dataset_path, 'users_TRAIN.csv'))
    headers = pd.read_csv(os.path.join(dataset_path, 'headers_TRAIN.csv'))
    img_info = pd.read_csv(os.path.join(dataset_path, 'img_info_TRAIN.csv'))
    popularity = pd.read_csv(os.path.join(dataset_path, 'popularity_TRAIN.csv'))

    dataset = process_meta_data(headers, img_info, popularity, path)
    print('Process meta data done!')

    process_user_data(dataset, users, path)
    print('Process user data done!')






