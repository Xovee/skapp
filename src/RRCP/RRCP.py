import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse

def load_model(model_path):
    model = torch.load(model_path, weights_only=False)
    model.cuda()
    model.eval()
    return model

def preprocess_RRCP_gold(input, dissembled_model_path, output, target_num, retrieval_num):
    batch_size = 128
    df = pd.read_pickle(input)
    
    (merged_text_vec_list, cls_vec_list, label_list, 
     retrieved_visual_feature_embedding_cls_list, retrieved_textual_feature_embedding_list, 
     retrieved_label_list) = (
        df['merged_text_vec'].tolist(), df['cls_vec'].tolist(), df['label'].tolist(), 
        df['retrieved_visual_feature_embedding_cls'].tolist(),
        df['retrieved_textual_feature_embedding'].tolist(), 
        df['retrieved_label_list'].tolist()
    )

    dissembled_model = load_model(dissembled_model_path)

    RRCP_gold_list_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size)):
            merged_text_vec = torch.tensor(merged_text_vec_list[i:i+batch_size], dtype=torch.float32).cuda()
            cls_vec = torch.tensor(cls_vec_list[i:i+batch_size], dtype=torch.float32).cuda()
            real_label = torch.tensor(label_list[i:i+batch_size], dtype=torch.float32).cuda()
            retrieved_visual_feature_embedding_cls = torch.tensor(retrieved_visual_feature_embedding_cls_list[i:i+batch_size], dtype=torch.float32).cuda()
            retrieved_textual_feature_embedding = torch.tensor(retrieved_textual_feature_embedding_list[i:i+batch_size], dtype=torch.float32).cuda()
            retrieved_label = torch.tensor(retrieved_label_list[i:i+batch_size], dtype=torch.float32).cuda()

            RRCP_gold_list = []

            for j in range(retrieval_num):
                retrieved_visual_feature = retrieved_visual_feature_embedding_cls[:, j, :, :]
                retrieved_textual_feature = retrieved_textual_feature_embedding[:, j, :, :]
                retrieved_label_feature = retrieved_label[:, j].unsqueeze(-1)

                label_without_retrieval = dissembled_model(cls_vec, merged_text_vec, cls_vec, merged_text_vec,
                                            retrieved_label_feature).squeeze(1)

                label_with_retrieval = dissembled_model(cls_vec, merged_text_vec,
                                                retrieved_visual_feature, retrieved_textual_feature,
                                                retrieved_label_feature).squeeze(1)

                RRCP_gold_list.append((abs(label_without_retrieval-real_label) - abs(label_with_retrieval-real_label)))

            RRCP_gold_list = torch.stack(RRCP_gold_list, dim=1).cpu().numpy().tolist()
            RRCP_gold_list_list.extend(RRCP_gold_list)

    df['RRCP_gold'] = RRCP_gold_list_list
    df.to_pickle(output)
    print('RRCP_gold processed and saved.')

def preprocess_RRCP_silver(input, dissembled_model_path, all_model_path, output, target_num, retrieval_num):
    batch_size = 128
    df = pd.read_pickle(input)
    
    (merged_text_vec_list, cls_vec_list, 
     retrieved_visual_feature_embedding_cls_list, retrieved_textual_feature_embedding_list, 
     retrieved_label_list) = (
        df['merged_text_vec'].tolist(), df['cls_vec'].tolist(), 
        df['retrieved_visual_feature_embedding_cls'].tolist(),
        df['retrieved_textual_feature_embedding'].tolist(), 
        df['retrieved_label_list'].tolist()
    )

    all_model = load_model(all_model_path)
    dissembled_model = load_model(dissembled_model_path)

    RRCP_silver_list_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size)):
            merged_text_vec = torch.tensor(merged_text_vec_list[i:i+batch_size], dtype=torch.float32).cuda()
            cls_vec = torch.tensor(cls_vec_list[i:i+batch_size], dtype=torch.float32).cuda()
            retrieved_visual_feature_embedding_cls = torch.tensor(retrieved_visual_feature_embedding_cls_list[i:i+batch_size], dtype=torch.float32).cuda()
            retrieved_textual_feature_embedding = torch.tensor(retrieved_textual_feature_embedding_list[i:i+batch_size], dtype=torch.float32).cuda()
            retrieved_label = torch.tensor(retrieved_label_list[i:i+batch_size], dtype=torch.float32).cuda()

            RRCP_silver_list = []

            retrieved_visual_feature_embedding_cls_ = retrieved_visual_feature_embedding_cls[:, :target_num, :, :]
            retrieved_textual_feature_embedding_ = retrieved_textual_feature_embedding[:, :target_num, :, :]
            retrieved_label_ = retrieved_label[:, :target_num]

            Predict = all_model(cls_vec, merged_text_vec, retrieved_visual_feature_embedding_cls_,
                                retrieved_textual_feature_embedding_, retrieved_label_).squeeze(-1)

            for j in range(retrieval_num):
                retrieved_visual_feature = retrieved_visual_feature_embedding_cls[:, j, :, :]
                retrieved_textual_feature = retrieved_textual_feature_embedding[:, j, :, :]
                retrieved_label_feature = retrieved_label[:, j].unsqueeze(-1)

                label_without_retrieval = dissembled_model(cls_vec, merged_text_vec, cls_vec, merged_text_vec,
                                            retrieved_label_feature).squeeze(1)

                label_with_retrieval = dissembled_model(cls_vec, merged_text_vec,
                                                retrieved_visual_feature, retrieved_textual_feature,
                                                retrieved_label_feature).squeeze(1)

                RRCP_silver_list.append((abs(Predict - label_without_retrieval) - abs(Predict - label_with_retrieval)))

            RRCP_silver_list = torch.stack(RRCP_silver_list, dim=1).cpu().numpy().tolist()
            RRCP_silver_list_list.extend(RRCP_silver_list)

    df['RRCP_silver'] = RRCP_silver_list_list
    df.to_pickle(output)
    print('RRCP_silver processed and saved.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some datasets.')
    parser.add_argument('--all_model_path', type=str, required=True, help='Path to all models')
    parser.add_argument('--dissembled_model_path', type=str, required=True, help='Path to dissembled models')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')

    args = parser.parse_args()

    target_num = 500
    all_model_path = args.all_model_path
    dissembled_model_path = args.dissembled_model_path

    retrieval_num = 500
    original_path = args.dataset_path

    for dataset in ['train', 'valid', 'test']:
        input_path = f'{original_path}/{dataset}.pkl'
        output_path = f'{original_path}/{dataset}.pkl'

        print(f'Processing {dataset} dataset...')
        # print('Processing RRCP_gold...')
        # preprocess_RRCP_gold(input_path, dissembled_model_path, output_path, target_num, retrieval_num)

        print('Processing RRCP_silver...')
        preprocess_RRCP_silver(input_path, dissembled_model_path, all_model_path, output_path, target_num,
                               retrieval_num)

        print(f'{dataset} dataset processing completed.')