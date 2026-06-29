import argparse
import io
import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import spacy
import torch
from angle_emb import AnglE
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor, ViTImageProcessor, ViTModel


DEFAULT_RAW_DIR = Path("datasets/raw_dataset/SMPD")
LEGACY_RAW_DIR = Path("datasets/origin_dataset/SMPD")
IMAGE_ZIP_NAMES = ("SMP_image_train_images.zip", "train_images.zip")


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def _batch_count(items, batch_size):
    return (len(items) + batch_size - 1) // batch_size


def _column_ready(dataframe, columns):
    return all(column in dataframe.columns and len(dataframe[column]) == len(dataframe) for column in columns)


def _candidate_raw_dirs():
    return [DEFAULT_RAW_DIR, LEGACY_RAW_DIR]


def _find_image_zip(image_zip=None):
    if image_zip:
        path = Path(image_zip)
        if not path.exists():
            raise FileNotFoundError(f"Image zip not found: {path}")
        return path

    for directory in _candidate_raw_dirs():
        for name in IMAGE_ZIP_NAMES:
            path = directory / name
            if path.exists():
                return path
    return None


def _find_image_root(image_root=None):
    if image_root:
        path = Path(image_root)
        if not path.exists():
            raise FileNotFoundError(f"Image root not found: {path}")
        return path

    for path in [Path("datasets/SMPD/pic"), DEFAULT_RAW_DIR, LEGACY_RAW_DIR]:
        if (
            path.exists()
            and ((path / "train").exists() or (path / "SMP_train_images").exists() or any(path.glob("*.jpg")))
        ):
            return path
    return None


class ImageStore:
    def __init__(self, image_root=None, image_zip=None):
        self.image_zip = _find_image_zip(image_zip) if image_zip or image_root is None else None
        self.image_root = None if self.image_zip else _find_image_root(image_root)
        self.zip_file = None

        if self.image_zip is None and self.image_root is None:
            raise FileNotFoundError("No SMPD image zip or image root was found")

    def __enter__(self):
        if self.image_zip:
            self.zip_file = zipfile.ZipFile(self.image_zip)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.zip_file:
            self.zip_file.close()

    def _zip_candidates(self, rel_path, image_id):
        rel_path = str(rel_path).replace("\\", "/")
        candidates = [rel_path]
        if rel_path.startswith("train/"):
            candidates.append(rel_path.replace("train/", "SMP_train_images/", 1))
        if image_id:
            candidates.append(f"{image_id}.jpg")
        return candidates

    def _root_candidates(self, rel_path, image_id):
        rel_path = Path(str(rel_path).replace("\\", os.sep))
        candidates = [self.image_root / rel_path]
        if image_id:
            candidates.append(self.image_root / f"{image_id}.jpg")
            candidates.append(self.image_root / "pic" / f"{image_id}.jpg")
        return candidates

    def open(self, rel_path, image_id=None):
        try:
            if self.zip_file:
                for candidate in self._zip_candidates(rel_path, image_id):
                    try:
                        with self.zip_file.open(candidate) as handle:
                            return Image.open(io.BytesIO(handle.read())).convert("RGB")
                    except KeyError:
                        continue
                return None

            for candidate in self._root_candidates(rel_path, image_id):
                if candidate.exists():
                    with Image.open(candidate) as image:
                        return image.convert("RGB")
            return None
        except Exception:
            return None


def _image_records(meta_data):
    rel_paths = (
        meta_data["image_rel_path"].astype(str).tolist()
        if "image_rel_path" in meta_data.columns
        else [f"{image_id}.jpg" for image_id in meta_data["image_id"].astype(str)]
    )
    return list(zip(meta_data["image_id"].astype(str).tolist(), rel_paths))


def image2text(meta_file_path, image_root=None, image_zip=None, batch_size=16, device=None, force=False):
    meta_data = pd.read_pickle(meta_file_path)
    if not force and _column_ready(meta_data, ["image_to_text"]):
        print("image_to_text already exists; skipping image captioning.")
        return 0

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def loading_model():
        model_name = "Salesforce/blip-image-captioning-large"
        processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        model.eval()
        return processor, model

    processor_text, model_text = loading_model()
    records = _image_records(meta_data)
    image_to_text_list = []
    missing_or_broken = 0

    with ImageStore(image_root, image_zip) as image_store:
        for _, batch_records in tqdm(_batched(records, batch_size), total=_batch_count(records, batch_size)):
            captions = ["0"] * len(batch_records)
            images = []
            image_positions = []

            for batch_pos, (image_id, rel_path) in enumerate(batch_records):
                image = image_store.open(rel_path, image_id)
                if image is None:
                    missing_or_broken += 1
                    continue
                images.append(image)
                image_positions.append(batch_pos)

            if images:
                inputs = processor_text(images=images, return_tensors="pt", padding=True).to(device)
                with torch.inference_mode():
                    generated_ids = model_text.generate(**inputs)
                decoded = processor_text.batch_decode(generated_ids, skip_special_tokens=True)

                for batch_pos, text in zip(image_positions, decoded):
                    captions[batch_pos] = text

            image_to_text_list.extend(captions)

    meta_data["image_to_text"] = image_to_text_list
    meta_data.to_pickle(meta_file_path)
    print(f"image2text done; missing or unreadable images: {missing_or_broken}")
    return 0


def image2vec(meta_file_path, image_root=None, image_zip=None, batch_size=64, device=None, force=False):
    meta_data = pd.read_pickle(meta_file_path)
    if not force and _column_ready(meta_data, ["mean_pooling_vec", "cls_vec"]):
        print("Image embeddings already exist; skipping image2vec.")
        return 0

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load_vit_model():
        model_name = "google/vit-base-patch16-224-in21k"
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name).to(device)
        model.eval()
        return processor, model

    processor_vec, model_vec = load_vit_model()
    records = _image_records(meta_data)
    mean_pooling_vec_list = []
    cls_vec_list = []
    missing_or_broken = 0

    def zero_embedding():
        return [[0.0] * 768]

    with ImageStore(image_root, image_zip) as image_store:
        for _, batch_records in tqdm(_batched(records, batch_size), total=_batch_count(records, batch_size)):
            mean_batch = [zero_embedding() for _ in batch_records]
            cls_batch = [zero_embedding() for _ in batch_records]
            images = []
            image_positions = []

            for batch_pos, (image_id, rel_path) in enumerate(batch_records):
                image = image_store.open(rel_path, image_id)
                if image is None:
                    missing_or_broken += 1
                    continue
                images.append(image)
                image_positions.append(batch_pos)

            if images:
                inputs = processor_vec(images=images, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model_vec(**inputs)

                cls_output = outputs.last_hidden_state[:, 0, :].detach().cpu()
                mean_pooling_output = outputs.last_hidden_state.mean(dim=1).detach().cpu()

                for encoded_pos, batch_pos in enumerate(image_positions):
                    mean_batch[batch_pos] = mean_pooling_output[encoded_pos:encoded_pos + 1].tolist()
                    cls_batch[batch_pos] = cls_output[encoded_pos:encoded_pos + 1].tolist()

            mean_pooling_vec_list.extend(mean_batch)
            cls_vec_list.extend(cls_batch)

    meta_data["mean_pooling_vec"] = mean_pooling_vec_list
    meta_data["cls_vec"] = cls_vec_list
    meta_data.to_pickle(meta_file_path)
    print(f"image2vec done; missing or unreadable images: {missing_or_broken}")
    return 0


def merged_text_create_and_to_vec(meta_file_path, batch_size=256, device=None, force=False):
    meta_data = pd.read_pickle(meta_file_path)
    if not force and _column_ready(meta_data, ["merged_text", "merged_text_vec"]):
        print("Merged text embeddings already exist; skipping text embedding.")
        return 0

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load_bert_model():
        model_name = "SeanLee97/angle-bert-base-uncased-nli-en-v1"
        angel = AnglE.from_pretrained(model_name, pooling_strategy="cls_avg")
        if hasattr(angel, "to"):
            angel = angel.to(device)
        elif str(device).startswith("cuda"):
            angel = angel.cuda()
        return angel

    text_list = meta_data["text"].fillna("").astype(str).tolist()
    image_to_text_list = meta_data["image_to_text"].fillna("").astype(str).tolist()
    merged_text_list = []
    merged_text_vec_list = []
    angel = load_bert_model()

    for _, batch_texts in tqdm(
        _batched([text + image_text for text, image_text in zip(text_list, image_to_text_list)], batch_size),
        total=_batch_count(text_list, batch_size),
    ):
        merged_text_list.extend(batch_texts)
        text_embedding = angel.encode(batch_texts, to_numpy=True)
        if text_embedding.ndim == 1:
            text_embedding = text_embedding.reshape(1, -1)
        for index in range(text_embedding.shape[0]):
            merged_text_vec_list.append(text_embedding[index:index + 1].tolist())

    meta_data["merged_text"] = merged_text_list
    meta_data["merged_text_vec"] = merged_text_vec_list
    meta_data.to_pickle(meta_file_path)
    return 0


def merged_text2nouns2verb2adj(meta_file_path, batch_size=256, force=False):
    meta_data = pd.read_pickle(meta_file_path)
    if not force and _column_ready(meta_data, ["nouns", "verbs", "adjectives"]):
        print("POS features already exist; skipping spaCy extraction.")
        return 0

    nlp = spacy.load("en_core_web_sm")
    n_nouns = []
    n_verbs = []
    n_adjectives = []
    merged_text_list = meta_data["merged_text"].fillna("").astype(str).tolist()

    for doc in tqdm(nlp.pipe(merged_text_list, batch_size=batch_size), total=len(merged_text_list)):
        n_nouns.append([token.text for token in doc if token.pos_ == "NOUN"])
        n_verbs.append([token.text for token in doc if token.pos_ == "VERB"])
        n_adjectives.append([token.text for token in doc if token.pos_ == "ADJ"])

    meta_data["nouns"] = n_nouns
    meta_data["verbs"] = n_verbs
    meta_data["adjectives"] = n_adjectives
    meta_data.to_pickle(meta_file_path)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Extract SMPD image, text, and POS features.")
    parser.add_argument("--dataset_path", default="datasets/SMPD/dataset.pkl", help="SMPD dataset pickle path")
    parser.add_argument("--image_root", default=None, help="Directory containing SMPD train images")
    parser.add_argument("--image_zip", default=None, help="Zip file containing SMPD train images")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu")
    parser.add_argument("--caption_batch_size", default=16, type=int, help="BLIP captioning batch size")
    parser.add_argument("--image_batch_size", default=64, type=int, help="ViT image embedding batch size")
    parser.add_argument("--text_batch_size", default=256, type=int, help="AnglE text embedding batch size")
    parser.add_argument("--spacy_batch_size", default=256, type=int, help="spaCy pipe batch size")
    parser.add_argument("--force", action="store_true", help="Recompute features even if output columns exist")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    image2text(
        args.dataset_path,
        args.image_root,
        args.image_zip,
        args.caption_batch_size,
        args.device,
        args.force,
    )
    image2vec(
        args.dataset_path,
        args.image_root,
        args.image_zip,
        args.image_batch_size,
        args.device,
        args.force,
    )
    merged_text_create_and_to_vec(args.dataset_path, args.text_batch_size, args.device, args.force)
    print("merged_text_create_and_to_vec done")
    merged_text2nouns2verb2adj(args.dataset_path, args.spacy_batch_size, args.force)
    print("merged_text2nouns2verb2adj done")
    print(f"Runtime: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
