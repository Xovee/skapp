import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # uncomment this line if you need to use a mirror
import time

from PIL import Image
import pandas as pd
import torch
from angle_emb import AnglE
from tqdm import tqdm
from transformers import (
    BartModel,
    BartTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
    ViTImageProcessor,
    ViTModel,
)
import spacy


def image2text(meta_file_path, image_path):
    # generate image captions using BLIP model and store them in the metadata file

    def loading_model():
        model_name = "Salesforce/blip-image-captioning-large"
        processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to("cuda")

        return processor, model


    def convert_image_to_text(processor, model, image_path):
        # load image and generate caption
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        text = processor.decode(out[0], skip_special_tokens=True)

        return text

    processor_text, model_text = loading_model()
    meta_data = pd.read_pickle(meta_file_path)
    pic_name_list = meta_data["image_id"]
    image_to_text_list = []

    for i in tqdm(range(len(pic_name_list))):
        current_id = pic_name_list[i]
        path = os.path.join(image_path, f"{current_id}.jpg")

        # handle missing images
        if not os.path.exists(path):
            text = "0"
        else:
            text = convert_image_to_text(processor_text, model_text, path)

        image_to_text_list.append(text)

    meta_data['image_to_text'] = image_to_text_list
    meta_data.to_pickle(meta_file_path)

    return 0


def image2vec(meta_file_path, image_path):
    # extract image embeddings using a pre-trained ViT model and save to metadata

    def load_VIT_model():
        model_name = "google/vit-base-patch16-224-in21k"
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name).to("cuda")

        return processor, model

    def image_to_embedding(processor, model, image_path):
        # get image embeddings from ViT
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token representation
        mean_pooling_output = torch.mean(outputs.last_hidden_state, dim=1)

        return mean_pooling_output, cls_output

    meta_data = pd.read_pickle(meta_file_path)
    pic_name_list = meta_data["image_id"]

    processor_vec, model_vec = load_VIT_model()
    mean_pooling_vec_list = []
    cls_vec_list = []

    batch_size = 1000  # batch size

    for i in tqdm(range(0, len(pic_name_list), batch_size)):
        batch_paths = pic_name_list[i:i+batch_size]

        for path in batch_paths:
            current_path = os.path.join(image_path, f"{path}.jpg")

            # handle missing images with zero vectors
            if not os.path.exists(current_path):
                mean_pooling_output = torch.zeros(1, 768).to("cuda")
                cls_output = torch.zeros(1, 768).to("cuda")
            else:
                mean_pooling_output, cls_output = image_to_embedding(processor_vec, model_vec, current_path)

            mean_pooling_vec_list.append(mean_pooling_output.tolist())
            cls_vec_list.append(cls_output.tolist())

            del mean_pooling_output
            del cls_output

    meta_data['mean_pooling_vec'] = mean_pooling_vec_list
    meta_data['cls_vec'] = cls_vec_list
    meta_data.to_pickle(meta_file_path)

    return 0


def merged_text_create_and_to_vec(meta_file_path):
    # merge usre-provided text with image captions and convert to embeddings using AnglE.

    def load_bert_model():
        model_name = "SeanLee97/angle-bert-base-uncased-nli-en-v1"
        angel = AnglE.from_pretrained(model_name, pooling_strategy='cls_avg').cuda()

        return angel

    meta_data = pd.read_pickle(meta_file_path)
    text_list = meta_data["text"]
    image_to_text_list = meta_data["image_to_text"]

    merged_text_list = []
    merged_text_vec_list = []

    angel = load_bert_model()

    for i in tqdm(range(len(text_list))):
        merged_text = text_list[i] + image_to_text_list[i]  # concatenate text and image caption
        merged_text_list.append(merged_text)

        text_to_embedding_result = angel.encode(merged_text, to_numpy=True)
        merged_text_vec_list.append(text_to_embedding_result.tolist())


    meta_data['merged_text'] = merged_text_list
    meta_data['merged_text_vec'] = merged_text_vec_list
    meta_data.to_pickle(meta_file_path)

    return 0


def merged_text2nouns2verb2adj(meta_file_path):
    # extract nouns, verbs, and adjectives from the merged text using spaCy
    spacy.cli.download("en_core_web_sm")  # download language model if not already present
    nlp = spacy.load("en_core_web_sm")

    n_nouns = []
    n_verbs = []
    n_adjectives = []

    meta_data = pd.read_pickle(meta_file_path)
    merged_text_list = meta_data["merged_text"]

    for i in tqdm(range(len(merged_text_list))):
        merged_text = merged_text_list[i]
        doc = nlp(merged_text)

        # extract POS-specific tokens
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        verbs = [token.text for token in doc if token.pos_ == "VERB"]
        adjectives = [token.text for token in doc if token.pos_ == "ADJ"]

        n_nouns.append(nouns)
        n_verbs.append(verbs)
        n_adjectives.append(adjectives)

    meta_data['nouns'] = n_nouns
    meta_data['verbs'] = n_verbs
    meta_data['adjectives'] = n_adjectives
    meta_data.to_pickle(meta_file_path)

    return 0


def main():
    start_time = time.time()

    # set path
    dataset_path = r"datasets/ICIP/dataset.pkl"
    image_path = r'datasets/origin_dataset/ICIP/pic'

    # sequentially process the dataset
    image2text(dataset_path, image_path)
    print("[1] Image to text complete.")

    image2vec(dataset_path, image_path)
    print("[2] Image to vec complete.")

    merged_text_create_and_to_vec(dataset_path)
    print("[3] Merge text and convert to vec complete.")

    merged_text2nouns2verb2adj(dataset_path)
    print("[4] Extract nouns, verbs, and adjectives complete.")

    # display total runtime
    print(f"Runtime: {(time.time() - start_time) / 60:.2f} minutes.")


if __name__ == '__main__':
    main()
