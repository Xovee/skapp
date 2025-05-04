import math
import os
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def encode_tags(word_list):
    # encodes tags into unique integers
    word_dict = {}
    encoded_list = []

    for sublist in word_list:

        if sublist == []:
            encoded_list.append([0])
            continue

        # handle string representation of list
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
    # merge metadata from different sources on "FlickrId"
    merged_data_1 = pd.merge(headers, img_info, on='FlickrId')
    merged_data = pd.merge(merged_data_1, popularity, on='FlickrId')

    # compute label using a log-based popularity metric
    merged_data['label'] = [math.log2(merged_data['Day30'][i] / 30 + 1) for i in range(len(merged_data['Day30']))]

    # encode tags as integers
    tags = encode_tags(merged_data['Tags'])

    # concatenate title and description
    title = merged_data['Title']
    description = merged_data['Description']
    text = [str(x) + ' ' + str(y) for x, y in zip(title, description)]

    # remove HTML anchor tags from text
    pattern = r'<a[^>]*>(.*?)</a>'
    clean_text = [re.sub(pattern, '', item) for item in text]

    # convert group info to integers
    avg_group_members = [int(item) for item in merged_data['AvgGroupsMemb']]
    avg_group_photos = [int(item) for item in merged_data['AvgGroupPhotos']]

    # build dataset dictionary
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

    # save processed dataset to file
    df = pd.DataFrame(dataset)
    df.to_pickle(path)

    return df


def process_user_data(dataset, users, path):
    # match user info with dataset entries
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

    # convert mean views to integers
    mean_views_list = [int(item) for item in mean_views_list]

    # add user info to dataset
    dataset['is_pro'] = is_pro_list
    dataset['has_status'] = has_status_list
    dataset['contacts'] = contacts_list
    dataset['photo_count'] = photo_count_list
    dataset['mean_views'] = mean_views_list

    # save updated dataset
    dataset.to_pickle(path)

    return dataset


def main():
    start_time = time.time()

    # load raw data
    dataset_path = Path("datasets/raw_dataset/ICIP")
    users = pd.read_csv(os.path.join(dataset_path / 'users_TRAIN.csv'))
    headers = pd.read_csv(os.path.join(dataset_path / 'headers_TRAIN.csv'))
    img_info = pd.read_csv(os.path.join(dataset_path / 'img_info_TRAIN.csv'))
    popularity = pd.read_csv(os.path.join(dataset_path / 'popularity_TRAIN.csv'))

    # proces metadata and user data
    path = Path("datasets/ICIP/dataset.pkl")
    dataset = process_meta_data(headers, img_info, popularity, path)
    print('[1] Process meta data complete.')

    process_user_data(dataset, users, path)
    print('[2] Process user data complete.')

    # display total runtime
    print(f"Runtime: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()
    
