import pandas as pd
from tqdm import tqdm
import os

def disassemble(path, output_path, retrieval_num):
    dataset = pd.read_pickle(path)

    label = dataset['label']
    mean_pooling_vec = dataset['mean_pooling_vec']
    merge_text_vec = dataset['merged_text_vec']
    retrieved_visual_feature_embedding_cls = dataset['retrieved_visual_feature_embedding_cls']
    retrieved_textual_feature_embedding = dataset['retrieved_textual_feature_embedding']
    retrieved_label_list = dataset['retrieved_label_list']

    disassembled_label = []
    disassembled_mean_pooling_vec = []
    disassembled_merge_text_vec = []
    disassembled_retrieved_visual_feature_embedding_cls = []
    disassembled_retrieved_textual_feature_embedding = []
    disassembled_retrieved_label_list = []
    disassembled_retrieved_textual_feature_embedding_list = []
    disassembled_retrieved_visual_feature_embedding_cls_list = []
    disassembled_retrieved_label_list_list = []

    for i in tqdm(range(len(label))):
        for j in range(retrieval_num):
            disassembled_label.append(label[i])
            disassembled_mean_pooling_vec.append(mean_pooling_vec[i])
            disassembled_merge_text_vec.append(merge_text_vec[i])
            disassembled_retrieved_visual_feature_embedding_cls.append(retrieved_visual_feature_embedding_cls[i][j])
            disassembled_retrieved_textual_feature_embedding.append(retrieved_textual_feature_embedding[i][j])
            disassembled_retrieved_label_list.append([retrieved_label_list[i][j]])
            disassembled_retrieved_visual_feature_embedding_cls_list.append(retrieved_visual_feature_embedding_cls[i])
            disassembled_retrieved_textual_feature_embedding_list.append(retrieved_textual_feature_embedding[i])
            disassembled_retrieved_label_list_list.append(retrieved_label_list[i])

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
    
    disassembled_dataset = disassembled_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    disassembled_dataset.to_pickle(output_path)
    print('Disassemble done!')


if __name__ == "__main__":
    retrieval_num = 500

    source_path = r'datasets/INS'
    disassemble_path = r'datasets/INS_dissembled'

    os.makedirs(disassemble_path, exist_ok=True)

    disassemble(os.path.join(source_path, 'train.pkl'), os.path.join(disassemble_path, 'train.pkl'), retrieval_num)
    disassemble(os.path.join(source_path, 'valid.pkl'), os.path.join(disassemble_path, 'valid.pkl'), retrieval_num)
    disassemble(os.path.join(source_path, 'test.pkl'), os.path.join(disassemble_path, 'test.pkl'), retrieval_num)
