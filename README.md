# SKAPP: Improving Multimodal Social media Popularity Prediction via Selective Retrieval Knowledge Augmentation

## Environmental Settings

Our experiments are conducted on Ubuntu 22.04, a single NVIDIA 3090Ti GPU, 128GB RAM, and Intel  i7-13700KF. SKAPP is implemented by `Python 3.9`, `Cuda 12.2`.

Create a virtual environment and install GPU-support packages via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name SKAPP python=3.9

# activate virtual environment
conda activate SKAPP

# install other dependencies
pip install -r requirements.txt
```

## Prepare

The storage format of the dataset and pre-trained model is:
```
project_root/
│
├── dataset/
 └── origin_dataset/
 ├── ICIP/
 │ ├── headers_TRAIN.csv
 │ ├── img_info_TRAIN.csv
 │ ├── popularity_TRAIN.csv
 │ ├── users_TRAIN.csv
 │ └── pic/
 │ ├── 1.jpg
 │ └── 2.jpg
 │
 ├── SMPD/
 │ ├── train_additional_information.json
 │ ├── train_category.json
 │ ├── train_temporalspatial_information.json
 │ ├── train_user_data.json
 │ ├── train_label.txt
 │ ├── train_text.json
 │ └── pic/
 │ ├── 1.jpg
 │ └── 2.jpg
 │
 └── INSTAGRAM/
 ├── json/
 │ ├── 1.json
 │ └── 2.json
 ├── pic/
 │ ├── 1.jpg
 │ └── 2.jpg
 └── post_info.txt

```
Download Required Datasets:

ICIP: http://www.visiongarage.altervista.org/popularitydataset/

SMPD: https://smp-challenge.com/download.html

INSTAGRAM: https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset

Place the datasets in the dataset/origin_dataset folder.

## Usage

Here we take the sample of ICIP dataset as an example to demonstrate the usage.

### Preprocess

```shell
cd SKAPP
python src/preprocess/ICIP/1_build_dataset.py
python src/preprocess/ICIP/2_preprocess.py
python src/preprocess/ICIP/3_retrieval.py
python src/preprocess/ICIP/4_disassemble.py
```

### Pre-training

```shell
cd SKAPP/src/RRCP
python train_all_item.py \
  --seed=2024 \
  --device=cuda:0 \
  --metric=MSE \
  --save=RESULT \
  --epochs=1000 \
  --batch_size=16 \
  --early_stop_turns=10 \
  --loss=MSE \
  --optim=Adam \
  --lr=1e-4 \
  --decay_rate=1.0 \
  --dataset_id=ICIP \
  --dataset_path=..\..\datasets \
  --retrieval_num=500 \
  --model_id=SKAPP_ALL_ITEMS
  
 python train_single_item.py \
  --seed=2024 \
  --device=cuda:0 \
  --metric=MSE \
  --save=RESULT \
  --epochs=1000 \
  --batch_size=1024 \
  --early_stop_turns=10 \
  --loss=MSE \
  --optim=Adam \
  --lr=1e-4 \
  --decay_rate=1.0 \
  --dataset_id=ICIP_dissembled \
  --dataset_path=..\..\datasets \
  --retrieval_num=1 \
  --model_id=SKAPP_SINGLE_ITEMS
```

Here we train the model `SKAPP_ALL_ITEMS` and `SKAPP_SINGLE_ITEMS` respectively. The model parameters will be saved in the path `./RESULT/`.

### Evaluation

Step 1: SKAPP generates

```shell
cd SKAPP
python src/RRCP/RRCP.py --all_model_path "path/to/all_model_path" --dissembled_model_path "path/to/dissembled_model_path" --dataset_path datasets/ICIP
```

Step 2: train SKAPP

```shell
cd SKAPP/src
python train.py \
  --seed=2024 \
  --device=cuda:0 \
  --metric=MSE \
  --save=RESULT \
  --epochs=1000 \
  --batch_size=64 \
  --early_stop_turns=5 \
  --loss=MSE \
  --optim=Adam \
  --lr=1e-4 \
  --decay_rate=1.0 \
  --dataset_id=ICIP \
  --dataset_path=..\datasets \
  --retrieval_num=500 \
  --model_id=SKAPP \
  --threshold_of_RRCP=0
```

Here we train the model `SKAPP` with the fine-tuning model. The model parameters will be saved in the path `./RESULT/`.

Step 3: evaluation

Replace the path `model_path` with the model parameter path obtained by training the code `python train.py` above.

```shell
cd SKAPP/src
python test.py \
  --seed=2024 \
  --device=cuda:0 \
  --metric='MSE,SRC,MAE' \
  --save=RESULT \
  --batch_size=256 \
  --dataset_id=ICIP \
  --dataset_path=..\datasets \
  --model_id=graph \
  --retrieval_num=500 \
  --model_path="path/to/model_path" \
  --threshold_of_RRCP=0
```









