
import pandas as pd
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os


def encode_tags_list(word_list):
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

def encode_tags(word_list):
    word_dict = {}
    encoded_list = []

    for sublist in word_list:

        if sublist == []:
            encoded_list.append([0])
            continue

        words = []

        if sublist not in word_dict.keys():
            word_dict[sublist] = len(word_dict) + 1

        words.append(word_dict[sublist])
        encoded_list.append(words)

    return encoded_list


def process_meta_data(train_text, train_additional_information, train_category, train_temporalspatial_information, train_user_data, label_data, path):
    



    all_tags = [tags.split() for tags in train_text['Alltags']]

    text = train_text['Title']

    pathalias = encode_tags(train_additional_information['Pathalias'])

    image_id = [f"{x}_{y}" for x, y in zip(train_additional_information['Uid'], train_additional_information['Pid'])]

    user_id = encode_tags(train_additional_information['Uid'])

    category = encode_tags(train_category['Category'])

    subcategory = encode_tags(train_category['Subcategory'])

    concepts = encode_tags(train_category['Concept'])

    postdate = train_temporalspatial_information['Postdate']

    photo_firstdate = train_user_data['photo_firstdate']

    photo_firstdatetaken = train_user_data['photo_firstdatetaken']

    photo_count = train_user_data['photo_count']

    time_zone_id = train_user_data['timezone_timezone_id']

    time_zone_offset = train_user_data['timezone_offset']

    label = label_data[0].tolist()

    dataset = {
        'image_id': image_id,
        'text': text,
        'tags': all_tags,
        'label': label,
        'user_id': user_id,
        'pathalias': pathalias,
        'category': category,
        'subcategory': subcategory,
        'concepts': concepts,
        'postdate': postdate,
        'photo_firstdate': photo_firstdate,
        'photo_firstdatetaken': photo_firstdatetaken,
        'photo_count': photo_count,
        'time_zone_id': time_zone_id,
        'time_zone_offset': time_zone_offset
    }

    df = pd.DataFrame(dataset)

    df.to_pickle(path)

    return df

if __name__ == "__main__":

    path = r"datasets/SMPD/dataset.pkl"
    origin_data_path = r"datasets/origin_dataset/SMPD"

    train_additional_information = pd.read_json(os.path.join(origin_data_path, 'train_additional_information.json'))
    train_category = pd.read_json(os.path.join(origin_data_path, 'train_category.json'))
    train_temporalspatial_information = pd.read_json(
        os.path.join(origin_data_path, 'train_temporalspatial_information.json'))
    train_user_data = pd.read_json(os.path.join(origin_data_path, 'train_user_data.json'))
    label_data = pd.read_csv(os.path.join(origin_data_path, 'train_label.txt'), header=None)
    train_text = pd.read_json(os.path.join(origin_data_path, 'train_text.json'))

    process_meta_data(train_text, train_additional_information, train_category, train_temporalspatial_information, train_user_data, label_data, path)
    print("Meta data processed!")




