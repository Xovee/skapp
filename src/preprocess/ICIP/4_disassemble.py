import os
import time

import pandas as pd
from tqdm import tqdm


def disassemble(path, output_path, retrieval_num):
    # load datasets from pickle files
    dataset = pd.read_pickle(path)

    # extract relevant fields
    label = dataset['label']
    mean_pooling_vec = dataset['mean_pooling_vec']
    merge_text_vec = dataset['merged_text_vec']
    retrieved_visual_feature_embedding_cls = dataset['retrieved_visual_feature_embedding_cls']
    retrieved_textual_feature_embedding = dataset['retrieved_textual_feature_embedding']
    retrieved_label_list = dataset['retrieved_label_list']

    # initialize lists for disassembled samples
    disassembled_label = []
    disassembled_mean_pooling_vec = []
    disassembled_merge_text_vec = []
    disassembled_retrieved_visual_feature_embedding_cls = []
    disassembled_retrieved_textual_feature_embedding = []
    disassembled_retrieved_label_list = []
    disassembled_retrieved_textual_feature_embedding_list = []
    disassembled_retrieved_visual_feature_embedding_cls_list = []
    disassembled_retrieved_label_list_list = []

    # for each sample, disassemble its retrievals into individual rows
    for i in tqdm(range(len(label))):
        label_i = label[i]
        mean_vec_i = mean_pooling_vec[i]
        text_vec_i = merge_text_vec[i]
        vis_feats_i = retrieved_visual_feature_embedding_cls[i]
        text_feats_i = retrieved_textual_feature_embedding[i]
        labels_i = retrieved_label_list[i]

        for j in range(retrieval_num):
            disassembled_label.append(label_i)
            disassembled_mean_pooling_vec.append(mean_vec_i)
            disassembled_merge_text_vec.append(text_vec_i)
            disassembled_retrieved_visual_feature_embedding_cls.append(vis_feats_i[j])
            disassembled_retrieved_textual_feature_embedding.append(text_feats_i[j])
            disassembled_retrieved_label_list.append([labels_i[j]])  # single retrieved label
            disassembled_retrieved_visual_feature_embedding_cls_list.append(vis_feats_i)
            disassembled_retrieved_textual_feature_embedding_list.append(text_feats_i)
            disassembled_retrieved_label_list_list.append(labels_i)

    # combine disassembled data into a new DataFrame
    disassembled_dataset = pd.DataFrame({
        'label': disassembled_label,
        'mean_pooling_vec': disassembled_mean_pooling_vec,
        'merged_text_vec': disassembled_merge_text_vec,
        'retrieved_visual_feature_embedding_cls': disassembled_retrieved_visual_feature_embedding_cls,
        'retrieved_textual_feature_embedding': disassembled_retrieved_textual_feature_embedding,
        'retrieved_label_list': disassembled_retrieved_label_list,
        'retrieved_visual_feature_embedding_cls_list': disassembled_retrieved_visual_feature_embedding_cls_list,
        'retrieved_textual_feature_embedding_list': disassembled_retrieved_textual_feature_embedding_list,
        'retrieved_label_list_list': disassembled_retrieved_label_list_list
    })
    
    # shuffle the dataset to avoid any order bias
    disassembled_dataset = disassembled_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # save to pickle file
    disassembled_dataset.to_pickle(output_path)

    print("Disassembled dataset saved successfully.")


def main():
    start_time = time.time()

    retrieval_num = 500  # number of retrievals per UGC

    source_path = r'datasets/ICIP'
    disassemble_path = r'datasets/ICIP_dissembled'

    os.makedirs(disassemble_path, exist_ok=True)

    # process and disassemble each split
    disassemble(os.path.join(source_path, 'train.pkl'), os.path.join(disassemble_path, 'train.pkl'), retrieval_num)
    print("[1] Disassemble train dataset complete.")
    disassemble(os.path.join(source_path, 'valid.pkl'), os.path.join(disassemble_path, 'valid.pkl'), retrieval_num)
    print("[2] Disassemble valid dataset complete.")
    disassemble(os.path.join(source_path, 'test.pkl'), os.path.join(disassemble_path, 'test.pkl'), retrieval_num)
    print("[3] Disassemble test dataset complete.")

    # display runtime
    print(f"Runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
