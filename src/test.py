import argparse
import os
from datetime import datetime
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error,mean_squared_error
from dataset import MyData, custom_collate_fn
import random
import numpy as np
from scipy.stats import spearmanr
from RRCP_prediction_variable_lenth import RRCP_prediction as my_model


BLUE = '\033[94m'
ENDC = '\033[0m'

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
    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_path} ")
    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")
    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")
    logger.info(BLUE + "Testing Starts!" + ENDC)


def delete_special_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace(BLUE, '')
    content = content.replace(ENDC, '')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def test(args):
    device = torch.device(args.device)
    model_id = args.model_id
    dataset_id = args.dataset_id
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"test_{model_id}_{dataset_id}_{timestamp}"
    father_folder_name = args.save
    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name)
    folder_path = os.path.join(father_folder_name, folder_name)
    os.mkdir(folder_path)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    test_data = MyData(args.retrieval_num, os.path.join(os.path.join(args.dataset_path, args.dataset_id, 'test.pkl')))
    test_data_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    model = my_model(retrieval_num=args.retrieval_num, threshold_of_RRCP=args.threshold_of_RRCP)
    model = model.to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    total_test_step = 0
    total_MAE = 0
    total_MSE = 0
    total_SRC = 0
    print_init_msg(logger, args)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc='Testing'):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
                retrieved_textual_feature_embedding, retrieved_label_list, RRCP, label = batch

            label = label.type(torch.float32)

            output = model.forward(mean_pooling_vec, merge_text_vec, retrieved_visual_feature_embedding_cls, \
                                   retrieved_textual_feature_embedding, retrieved_label_list, RRCP)

            output = output.to('cpu')
            label = label.to('cpu')
            output = np.array(output)
            label = np.array(label)

            MAE = mean_absolute_error(label, output)
            SRC, _ = spearmanr(output, label)
            MSE = mean_squared_error(y_pred=output, y_true=label)
            total_test_step += 1
            total_MAE += MAE
            total_SRC += SRC
            total_MSE += MSE

    logger.warning(f"[ Test Result ]:  \n {args.metric[0]} = {total_MSE / total_test_step}"
                   f"\n{args.metric[1]} = {total_SRC / total_test_step}\n{args.metric[2]} = {total_MAE / total_test_step}\n")
    logger.info("Test is ended!")
    delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")


def main(args):
    seed_init(args.seed)
    test(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default='2024', type=str, help='value of random seed')
    parser.add_argument('--device', default='cuda:0', type=str, help='device used in testing')
    parser.add_argument('--metric', default=['MSE', 'SRC', 'MAE'], type=list, help='the judgement of the testing')
    parser.add_argument('--save', default=r'RESULT', type=str, help='folder to save the results')
    parser.add_argument('--batch_size', default=256, type=int, help='training batch size')
    parser.add_argument('--dataset_id', default='ICIP', type=str, help='id of dataset')
    parser.add_argument('--dataset_path', default=r'../datasets', type=str, help='path of dataset')
    parser.add_argument('--model_id', default='SKAPP', type=str, help='id of model')
    parser.add_argument('--retrieval_num', default=500, type=int, help='number of retrieval')
    parser.add_argument('--model_path',
                        default=r"",
                        type=str, help='path of trained model')
    parser.add_argument('--threshold_of_RRCP', default=0, type=float)
    args = parser.parse_args()

    main(args)