# SKAPP

This repo contains a reference implementation of the SKAPP model described in the following paper:

> Xovee Xu, Yifan Zhang, Fan Zhou, and Jingkuan Song  
> Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation  
> AAAI Conference on Artificial Intelligence, 2025.  

SKAPP is a multimodal learning framework for social UGCs. It is equipped with a
meta retriever, a selective refiner, and a knowledge augmentation prediction
network.

## Environmental Settings

Our experiments are conducted on Ubuntu 22.04, a single NVIDIA 3090Ti GPU, 128GB
RAM, and Intel i7-13700KF.

Create a virtual environment and install GPU-support packages via
[Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name skapp python=3.9

# activate virtual environment
conda activate skapp

# install other dependencies
# make sure CUDA and PyTorch are installed for your system
pip3 install torch ..... # please refer to https://pytorch.org/ for your specific machine and software conditions
pip3 install pandas huggingface-hub tqdm scikit-learn transformers angle_emb
# install spacy: https://spacy.io/usage
```

## Dataset Preparation

First, download the datasets:

- ICIP: the original link (http://www.visiongarage.altervista.org/popularitydataset/) is currently unavailable.
  A read-only copy of the raw dataset is available on [Google Drive](https://drive.google.com/drive/folders/1eToci8-r0_E-zUuvbqSnAPkmO8aHvm67?usp=sharing) for academic research only.
- SMPD: https://smp-challenge.com
- Instagram: https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset

Then place the datasets in the corresponding `datasets/raw_dataset/` folder.

The storage format of the dataset is as follows:

```text
project_root/
datasets/
  raw_dataset/
    ICIP/
      headers_TRAIN.csv
      img_info_TRAIN.csv
      popularity_TRAIN.csv
      users_TRAIN.csv
      pic/
        1.jpg
        2.jpg
    SMPD/
      train_additional_information.json
      train_category.json
      train_temporalspatial_information.json
      train_user_data.json
      train_label.txt
      train_text.json
      pic/
        1.jpg
        2.jpg
    Instagram/
      json/
        1.json
        2.json
      pic/
        1.jpg
        2.jpg
      post_info.txt
```

## Usage

Here we take the ICIP dataset as an example to demonstrate the usage.

### Preprocess

Run the following commands for preprocessing the datasets. During preprocessing,
the pretrained models will be downloaded once.

```shell
cd skapp
python src/preprocess/ICIP/1_build_dataset.py
python src/preprocess/ICIP/2_preprocess.py
python src/preprocess/ICIP/3_retrieval.py
python src/preprocess/ICIP/4_disassemble.py  # prepares the dynamic single-item training split
```

### Pre-training

```shell
python src/RRCP/train_all_item.py --dataset_id=ICIP

python src/RRCP/train_single_item.py --dataset_id=ICIP_dissembled
```

Here we train the models `skapp_all_items` and `skapp_single_item`. The model
parameters will be saved in `./saved_models/`.

### Evaluation

Step 1: Obtain RRCP.

Remember to replace `"PATH"` with the actual saved model path, for example
`trained_model/model_10.pth`.

```shell
python src/RRCP/RRCP.py --all_model_path "PATH" --dissembled_model_path "PATH" --dataset_path datasets/ICIP
```

Step 2: Training.

```shell
python src/train.py --dataset_id=ICIP --model_id=SKAPP
```

Here we train the final `SKAPP` model. The model parameters will be saved in
`./saved_models/`.

Step 3: Evaluation.

Remember to replace the actual `model_path`, for example
`checkpoint_10_epoch.pkl`.

```shell
python src/test.py --dataset_id=ICIP --model_id=SKAPP --model_path="PATH"
```

### Hyper-Parameters

Please refer to `config.yaml`.

## Citation

```bibtex
@inproceedings{xu2025improving,
  title = {Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation},
  author = {Xovee Xu and Yifan Zhang and Fan Zhou and Jingkuan Song},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2025},
  volume = {39},
  number = {1},
  month = {apr},
  numpages = {9},
  pages = {932--940},
  publisher = {AAAI},
  doi = {10.1609/aaai.v39i1.32078}
}
```

## LICENSE

MIT
