import logging
import os
import argparse, sys
from datetime import datetime
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from config_utils import apply_cfg, explicit_arg_dests, load_cfg
from dataset import MyData, custom_collate_fn
from RRCP_prediction_variable_lenth import RRCP_prediction as my_model
import random
import numpy as np
from scipy.stats import spearmanr

BLUE = '\033[94m'
ENDC = '\033[0m'
DEFAULT_METADATA_FIELDS = {
    'ICIP': ['mean_views'],
}
DEFAULT_METADATA_TRANSFORMS = {
    'ICIP': 'log1p',
}


def parse_metadata_fields(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(field).strip() for field in value if str(field).strip()]
    return [field.strip() for field in str(value).split(',') if field.strip()]


def resolve_metadata_fields(args):
    if args.metadata_fields is None:
        args.metadata_fields = DEFAULT_METADATA_FIELDS.get(args.dataset_id, [])
        if args.metadata_transform is None:
            args.metadata_transform = DEFAULT_METADATA_TRANSFORMS.get(args.dataset_id, 'none')
    elif args.metadata_transform is None:
        args.metadata_transform = 'none'
    return args.metadata_fields


def unpack_batch(batch):
    if len(batch) == 8:
        mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
            retrieved_textual_feature_embedding, retrieved_label_list, RRCP, metadata, label = batch
    else:
        mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
            retrieved_textual_feature_embedding, retrieved_label_list, RRCP, label = batch
        metadata = None

    return mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
        retrieved_textual_feature_embedding, retrieved_label_list, RRCP, metadata, label


def seed_init(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_init_msg(logger, args):

    logger.info(BLUE + 'Random Seed: ' + ENDC + f"{args.seed} ")
    logger.info(BLUE + 'Device: ' + ENDC + f"{args.device} ")
    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_id} ")
    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")
    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")
    logger.info(BLUE + "Optimizer: " + ENDC + f"{args.optim}(lr = {args.lr})")
    logger.info(BLUE + "Total Epoch: " + ENDC + f"{args.epochs} Turns")
    logger.info(BLUE + "Retrieval Num: " + ENDC + f"{args.retrieval_num}")
    logger.info(BLUE + "Early Stop: " + ENDC + f"{args.early_stop_turns} Turns")
    logger.info(BLUE + "Batch Size: " + ENDC + f"{args.batch_size}")
    logger.info(BLUE + "Threshold of RRCP: " + ENDC + f"{args.threshold_of_RRCP}")
    logger.info(BLUE + "Metadata Fields: " + ENDC + f"{args.metadata_fields}")
    logger.info(BLUE + "Metadata Transform: " + ENDC + f"{args.metadata_transform}")
    logger.info(BLUE + "Training Starts!" + ENDC)


def make_saving_folder_and_logger(args, timestamp):
    folder_name = f"train_{args.model_id}_{args.dataset_id}_{args.retrieval_num}_{args.metric}_{timestamp}"

    parent_folder_name = args.save
    if not os.path.exists(parent_folder_name):
        os.makedirs(parent_folder_name)
    folder_path = os.path.join(parent_folder_name, folder_name)
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(f'{folder_path}/log.txt')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return parent_folder_name, folder_name, logger


def delete_model(parent_folder_name, folder_name, min_turn):
    model_name_list = os.listdir(f"{parent_folder_name}/{folder_name}")
    for i in range(len(model_name_list)):
        if model_name_list[i] != f'checkpoint_{min_turn}_epoch.pkl' and model_name_list[i] != 'log.txt':
            os.remove(os.path.join(f'{parent_folder_name}/{folder_name}', model_name_list[i]))

def force_stop(msg):
    print(msg)
    sys.exit(1)


def make_data_loader(dataset, args, device):
    loader_kwargs = {
        'batch_size': args.batch_size,
        'collate_fn': custom_collate_fn,
        'pin_memory': device.type == 'cuda',
        'num_workers': args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
    return DataLoader(dataset=dataset, **loader_kwargs)


def train_val(args):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    parent_folder_name, folder_name, logger = make_saving_folder_and_logger(args, timestamp)

    device = torch.device(args.device)

    train_data = MyData(
        args.retrieval_num,
        os.path.join(args.dataset_path, args.dataset_id, 'train.pkl'),
        metadata_fields=args.metadata_fields,
        metadata_transform=args.metadata_transform,
    )
    valid_data = MyData(
        args.retrieval_num,
        os.path.join(os.path.join(args.dataset_path, args.dataset_id, 'valid.pkl')),
        metadata_fields=args.metadata_fields,
        metadata_transform=args.metadata_transform,
    )
    train_data_loader = make_data_loader(train_data, args, device)
    valid_data_loader = make_data_loader(valid_data, args, device)

    model = my_model(
        retrieval_num=args.retrieval_num,
        threshold_of_RRCP=args.threshold_of_RRCP,
        metadata_dim=len(args.metadata_fields),
    )
    model = model.to(device)

    if args.loss == 'BCE':
        loss_fn = torch.nn.BCELoss()
    elif args.loss == 'MSE':
        loss_fn = torch.nn.MSELoss()
    else:
        force_stop('Invalid parameter loss!')

    loss_fn.to(device)
    if args.optim == 'Adam':
        optim = Adam(model.parameters(), args.lr)
    elif args.optim == 'SGD':
        optim = SGD(model.parameters(), args.lr)
    else:
        force_stop('Invalid parameter optim!')

    min_valid_loss = 1008611
    min_turn = 0

    print_init_msg(logger, args)
    for i in range(args.epochs):
        logger.info(f"-----------------------------------Epoch {i + 1} Start!-----------------------------------")
        train_loss, valid_loss = run_one_epoch(args, model, loss_fn, optim, train_data_loader,
                                               valid_data_loader,
                                               device)
        logger.info(f"[ Epoch {i + 1} (train) ]: avg_loss = {train_loss}")
        logger.info(f"[ Epoch {i + 1} (valid) ]: avg_loss = {valid_loss}")

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            min_turn = i + 1
        logger.critical(
            f"Current Best Valid Loss comes from Epoch {min_turn} , min_valid_loss = {min_valid_loss}")

        checkpoint = {"model_state_dict": model.state_dict()}
        path_checkpoint = f"{parent_folder_name}/{folder_name}/checkpoint_{i + 1}_epoch.pkl"

        torch.save(checkpoint, path_checkpoint)

        logger.info("Model has been saved successfully!")
        if (i + 1) - min_turn > args.early_stop_turns:
            break
    delete_model(parent_folder_name, folder_name, min_turn)
    logger.info(BLUE + "Training is ended!" + ENDC)

    if min_turn == 0:
        logger.warning("Training did not complete successfully. Deleting empty folder.")
        os.rmdir(os.path.join(parent_folder_name, folder_name))


def run_one_epoch(args, model, loss_fn, optim, train_data_loader, valid_data_loader, device):

    model.train()
    total_train_loss = 0.0
    train_sample_count = 0

    for batch in tqdm(train_data_loader, desc='Training Progress'):
        batch = [item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else item for item in batch]
        mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
            retrieved_textual_feature_embedding, retrieved_label_list, RRCP, metadata, label = unpack_batch(batch)

        target = label.type(torch.float32)

        output = model.forward(mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls,
                retrieved_textual_feature_embedding, retrieved_label_list, RRCP, metadata)

        loss = loss_fn(output, target)

        optim.zero_grad()
        loss.backward()
        optim.step()
        batch_size = target.size(0)
        total_train_loss += loss.item() * batch_size
        train_sample_count += batch_size

    model.eval()
    total_valid_loss = 0.0
    valid_sample_count = 0
    with torch.no_grad():
        for batch in tqdm(valid_data_loader, desc='Validating Progress'):
            batch = [item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else item for item in batch]

            mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
                retrieved_textual_feature_embedding, retrieved_label_list, RRCP, metadata, label = unpack_batch(batch)

            target = label.type(torch.float32)

            output = model.forward(mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls,
                                   retrieved_textual_feature_embedding, retrieved_label_list, RRCP, metadata)

            loss = loss_fn(output, target)

            batch_size = target.size(0)
            total_valid_loss += loss.item() * batch_size
            valid_sample_count += batch_size

    return total_train_loss / train_sample_count, total_valid_loss / valid_sample_count

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default='2024', type=str, help='value of random seed')
    parser.add_argument('--device', default='cuda:0', type=str, help='device used in training')
    parser.add_argument('--metric', default='MSE', type=str, help='the judgement of the training')
    parser.add_argument('--save', default=r'./saved_models',
                        type=str, help='folder to save the results')
    parser.add_argument('--epochs', default=1000, type=int, help='max number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--early_stop_turns', default=5, type=int, help='early stop turns of training')
    parser.add_argument('--loss', default='MSE', type=str, help='loss function, options: BCE, MSE')
    parser.add_argument('--optim', default='Adam', type=str, help='optim, options: SGD, Adam')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')
    parser.add_argument('--dataset_id', default='ICIP', type=str, help='id of dataset')
    parser.add_argument('--dataset_path', default=r'./datasets', type=str, help='path of dataset')
    parser.add_argument('--retrieval_num', default=50, type=int, help='number of retrieval')
    parser.add_argument('--model_id', default='SKAPP', type=str, help='id of model')
    parser.add_argument('--threshold_of_RRCP', default=0, type=float)
    parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--metadata_fields', default=None, type=parse_metadata_fields,
                        help='comma-separated metadata fields to use as additional predictors')
    parser.add_argument('--metadata_transform', default=None, choices=['none', 'log1p'],
                        help='transform applied to metadata fields')

    args = parser.parse_args()

    cfg = load_cfg(args.dataset_id, 'train')
    apply_cfg(parser, args, cfg, explicit_arg_dests(parser))
    resolve_metadata_fields(args)

    seed_init(args.seed)
    train_val(args)


if __name__ == '__main__':
    main()
