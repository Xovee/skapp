import pandas as pd
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re
import json
import os


def text2pkl(txt, path):

    txt.to_pickle(path)

def process_meta_data(json_path, path):

    meta_data = pd.read_pickle(path)

    all_pic_name_list = meta_data[4]
    all_user_data_json = meta_data[3]
    all_user_name_list = meta_data[1]

    image_id_list = []
    text_list = []
    comment_num_list = []
    label_list = []
    user_id_list = []
    taken_timestamp_list = []
    user_name_list = []

    for i in tqdm(range(len(meta_data))):

        if i % 7 != 0:

            continue

        user_data_json_path = all_user_data_json[i]

        pic_name_list = eval(all_pic_name_list[i])

        user_name = all_user_name_list[i]

        with open(os.path.join(r"/home/icdm/zyf/popularity/dataset/instgram/dataset/json", user_data_json_path), 'r',
                  encoding='UTF-8') as json_file:

            user_data = json.load(json_file)

            label = user_data["edge_media_preview_like"]["count"]

            edge_media_to_caption = user_data["edge_media_to_caption"]["edges"]

            if len(edge_media_to_caption) != 0:
                caption = edge_media_to_caption[0]["node"]["text"]
            else:
                caption = ""

            comment_num = user_data["edge_media_to_comment"]["count"]

            user_id = user_data["owner"]["id"]

            taken_at_timestamp = user_data["taken_at_timestamp"]

        for pic_name in pic_name_list:

            image_id_list.append(pic_name)

            label_list.append(label)

            text_list.append(caption)

            comment_num_list.append(comment_num)

            user_id_list.append(user_id)

            taken_timestamp_list.append(taken_at_timestamp)

            user_name_list.append(user_name)


    data = {

        "image_id": image_id_list,
        "text": text_list,
        "comment_num": comment_num_list,
        "label": label_list,
        "user_id": user_id_list,
        "taken_timestamp": taken_timestamp_list,
        "user_name": user_name_list

    }

    data_frame = pd.DataFrame(data)
    data_frame.to_pickle(path)


if __name__ == "__main__":

    path = r"datasets/origin_dataset/INS/dataset.pkl"
    origin_data_path = r"datasets/origin_dataset/INS"

    txt = pd.read_csv("datasets/origin_dataset/INS/post_info.txt")
    json_path = r"datasets/origin_dataset/INS/json"

    text2pkl(txt, path)
    dataset = process_meta_data(json_path, path)
    print('Process meta data done!')







