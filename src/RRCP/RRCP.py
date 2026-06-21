import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from pathlib import Path

def load_model(model_path):
    model = torch.load(model_path, weights_only=False)
    model.cuda()
    model.eval()
    return model


def _stack_feature(series):
    return np.asarray(series.tolist(), dtype=np.float32)


def _retrieved_labels(df):
    label_column = 'retrieved_label_list' if 'retrieved_label_list' in df.columns else 'retrieved_label'
    return np.asarray(df[label_column].tolist(), dtype=np.float32)


def _prepare_retrieval_source(df, input_path):
    if {'retrieved_visual_feature_embedding_cls', 'retrieved_textual_feature_embedding'}.issubset(df.columns):
        return {
            'mode': 'expanded',
            'visual': df['retrieved_visual_feature_embedding_cls'].tolist(),
            'textual': df['retrieved_textual_feature_embedding'].tolist(),
        }

    pool_path = Path(input_path).parent / 'retrieval_pool.pkl'
    if 'retrieved_item_id' not in df.columns or not pool_path.exists():
        raise KeyError(
            'Missing retrieved feature columns and feature-bank metadata. '
            'Expected retrieved_item_id plus retrieval_pool.pkl.'
        )

    retrieval_pool = pd.read_pickle(pool_path)
    id_to_pos = {item_id: idx for idx, item_id in enumerate(retrieval_pool['image_id'].tolist())}
    return {
        'mode': 'feature_bank',
        'id_lists': df['retrieved_item_id'].tolist(),
        'id_to_pos': id_to_pos,
        'visual_bank': _stack_feature(retrieval_pool['cls_vec']),
        'textual_bank': _stack_feature(retrieval_pool['merged_text_vec']),
    }


def _retrieved_batch(source, start, end, retrieval_num):
    if source['mode'] == 'expanded':
        visual = np.asarray(source['visual'][start:end], dtype=np.float32)[:, :retrieval_num]
        textual = np.asarray(source['textual'][start:end], dtype=np.float32)[:, :retrieval_num]
    else:
        indices = np.asarray(
            [[source['id_to_pos'][item_id] for item_id in item_ids[:retrieval_num]]
             for item_ids in source['id_lists'][start:end]],
            dtype=np.int64,
        )
        visual = source['visual_bank'][indices]
        textual = source['textual_bank'][indices]
    return torch.from_numpy(visual).cuda(), torch.from_numpy(textual).cuda()


def _predict_single_item_delta(model, current_visual, current_textual, retrieved_visual,
                               retrieved_textual, retrieved_label, chunk_size=8192):
    batch_size, retrieval_num = retrieved_label.shape
    flat_label = retrieved_label.reshape(batch_size * retrieval_num, 1)
    current_visual_flat = current_visual.unsqueeze(1).expand(
        -1, retrieval_num, *current_visual.shape[1:]
    ).reshape(batch_size * retrieval_num, *current_visual.shape[1:])
    current_textual_flat = current_textual.unsqueeze(1).expand(
        -1, retrieval_num, *current_textual.shape[1:]
    ).reshape(batch_size * retrieval_num, *current_textual.shape[1:])
    retrieved_visual_flat = retrieved_visual.reshape(batch_size * retrieval_num, *retrieved_visual.shape[2:])
    retrieved_textual_flat = retrieved_textual.reshape(batch_size * retrieval_num, *retrieved_textual.shape[2:])

    labels_without = []
    labels_with = []
    for start in range(0, flat_label.size(0), chunk_size):
        end = start + chunk_size
        label_chunk = flat_label[start:end]
        current_visual_chunk = current_visual_flat[start:end]
        current_textual_chunk = current_textual_flat[start:end]

        labels_without.append(
            model(current_visual_chunk, current_textual_chunk, current_visual_chunk,
                  current_textual_chunk, label_chunk).squeeze(-1)
        )
        labels_with.append(
            model(current_visual_chunk, current_textual_chunk,
                  retrieved_visual_flat[start:end], retrieved_textual_flat[start:end],
                  label_chunk).squeeze(-1)
        )

    return (
        torch.cat(labels_without, dim=0).reshape(batch_size, retrieval_num),
        torch.cat(labels_with, dim=0).reshape(batch_size, retrieval_num),
    )

def preprocess_RRCP_gold(input, dissembled_model_path, output, target_num, retrieval_num):
    batch_size = 128
    df = pd.read_pickle(input)

    merged_text_vec_list = df['merged_text_vec'].tolist()
    cls_vec_list = df['cls_vec'].tolist()
    label_list = df['label'].tolist()
    retrieved_label_list = _retrieved_labels(df)
    retrieval_source = _prepare_retrieval_source(df, input)

    dissembled_model = load_model(dissembled_model_path)

    RRCP_gold_list_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size)):
            end = min(i + batch_size, len(df))
            merged_text_vec = torch.from_numpy(np.asarray(merged_text_vec_list[i:end], dtype=np.float32)).cuda()
            cls_vec = torch.from_numpy(np.asarray(cls_vec_list[i:end], dtype=np.float32)).cuda()
            real_label = torch.from_numpy(np.asarray(label_list[i:end], dtype=np.float32)).cuda()
            retrieved_visual_feature_embedding_cls, retrieved_textual_feature_embedding = _retrieved_batch(
                retrieval_source, i, end, retrieval_num
            )
            retrieved_label = torch.from_numpy(retrieved_label_list[i:end, :retrieval_num]).cuda()

            label_without_retrieval, label_with_retrieval = _predict_single_item_delta(
                dissembled_model, cls_vec, merged_text_vec, retrieved_visual_feature_embedding_cls,
                retrieved_textual_feature_embedding, retrieved_label
            )
            RRCP_gold = torch.abs(label_without_retrieval - real_label.unsqueeze(1)) - torch.abs(
                label_with_retrieval - real_label.unsqueeze(1)
            )
            RRCP_gold_list_list.extend(RRCP_gold.cpu().numpy().tolist())

    df['RRCP_gold'] = RRCP_gold_list_list
    df.to_pickle(output)
    print('RRCP_gold processed and saved.')

def preprocess_RRCP_silver(input, dissembled_model_path, all_model_path, output, target_num, retrieval_num):
    batch_size = 128
    df = pd.read_pickle(input)

    merged_text_vec_list = df['merged_text_vec'].tolist()
    cls_vec_list = df['cls_vec'].tolist()
    retrieved_label_list = _retrieved_labels(df)
    retrieval_source = _prepare_retrieval_source(df, input)

    all_model = load_model(all_model_path)
    dissembled_model = load_model(dissembled_model_path)

    RRCP_silver_list_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size)):
            end = min(i + batch_size, len(df))
            merged_text_vec = torch.from_numpy(np.asarray(merged_text_vec_list[i:end], dtype=np.float32)).cuda()
            cls_vec = torch.from_numpy(np.asarray(cls_vec_list[i:end], dtype=np.float32)).cuda()
            retrieved_visual_feature_embedding_cls, retrieved_textual_feature_embedding = _retrieved_batch(
                retrieval_source, i, end, retrieval_num
            )
            retrieved_label = torch.from_numpy(retrieved_label_list[i:end, :retrieval_num]).cuda()

            target_count = min(target_num, retrieved_label.shape[1])
            retrieved_visual_feature_embedding_cls_ = retrieved_visual_feature_embedding_cls[:, :target_count, :, :]
            retrieved_textual_feature_embedding_ = retrieved_textual_feature_embedding[:, :target_count, :, :]
            retrieved_label_ = retrieved_label[:, :target_count]

            Predict = all_model(cls_vec, merged_text_vec, retrieved_visual_feature_embedding_cls_,
                                retrieved_textual_feature_embedding_, retrieved_label_).squeeze(-1)

            label_without_retrieval, label_with_retrieval = _predict_single_item_delta(
                dissembled_model, cls_vec, merged_text_vec, retrieved_visual_feature_embedding_cls,
                retrieved_textual_feature_embedding, retrieved_label
            )
            RRCP_silver = torch.abs(Predict.unsqueeze(1) - label_without_retrieval) - torch.abs(
                Predict.unsqueeze(1) - label_with_retrieval
            )
            RRCP_silver_list_list.extend(RRCP_silver.cpu().numpy().tolist())

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
