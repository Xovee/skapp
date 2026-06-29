import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd


DEFAULT_RAW_DIR = Path("datasets/raw_dataset/SMPD")
LEGACY_RAW_DIR = Path("datasets/origin_dataset/SMPD")
METADATA_ZIP_NAMES = ("SMP_image_train_metadata.zip", "train_allmetadata_json.zip")
METADATA_DIR_NAMES = ("train_allmetadata_json",)
TRAIN_FILES = {
    "additional": "train_additional_information.json",
    "category": "train_category.json",
    "temporal": "train_temporalspatial_information.json",
    "user": "train_user_data.json",
    "text": "train_text.json",
    "label": "train_label.txt",
    "image_path": "train_img_filepath.txt",
}


def _normalize_path(value):
    return str(value).replace("\\", "/").strip()


def _candidate_raw_dirs(raw_dir):
    raw_dir = Path(raw_dir)
    candidates = [raw_dir]
    if raw_dir != LEGACY_RAW_DIR:
        candidates.append(LEGACY_RAW_DIR)
    return candidates


def _find_metadata_zip(raw_dir, metadata_zip=None):
    if metadata_zip:
        path = Path(metadata_zip)
        if not path.exists():
            raise FileNotFoundError(f"Metadata zip not found: {path}")
        return path

    for directory in _candidate_raw_dirs(raw_dir):
        for name in METADATA_ZIP_NAMES:
            path = directory / name
            if path.exists():
                return path
    return None


def _find_metadata_dir(raw_dir):
    for directory in _candidate_raw_dirs(raw_dir):
        if all((directory / name).exists() for name in TRAIN_FILES.values()):
            return directory
        for name in METADATA_DIR_NAMES:
            nested = directory / name
            if all((nested / file_name).exists() for file_name in TRAIN_FILES.values()):
                return nested
    return None


def _zip_member(zip_file, file_name):
    matches = [
        name for name in zip_file.namelist()
        if not name.startswith("__MACOSX/") and name.endswith(file_name)
    ]
    if not matches:
        raise FileNotFoundError(f"{file_name} not found in metadata zip")
    return sorted(matches, key=len)[0]


def _read_json_from_zip(zip_path, file_name):
    with zipfile.ZipFile(zip_path) as zip_file:
        with zip_file.open(_zip_member(zip_file, file_name)) as handle:
            return pd.read_json(handle)


def _read_lines_from_zip(zip_path, file_name):
    with zipfile.ZipFile(zip_path) as zip_file:
        with zip_file.open(_zip_member(zip_file, file_name)) as handle:
            return [
                line.decode("utf-8", errors="replace").strip()
                for line in handle
                if line.strip()
            ]


def _read_json_from_dir(metadata_dir, file_name):
    return pd.read_json(Path(metadata_dir) / file_name)


def _read_lines_from_dir(metadata_dir, file_name):
    with open(Path(metadata_dir) / file_name, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _read_metadata(raw_dir, metadata_zip=None):
    zip_path = _find_metadata_zip(raw_dir, metadata_zip)
    if zip_path:
        print(f"Reading SMPD metadata from {zip_path}")
        read_json = lambda file_name: _read_json_from_zip(zip_path, file_name)
        read_lines = lambda file_name: _read_lines_from_zip(zip_path, file_name)
    else:
        metadata_dir = _find_metadata_dir(raw_dir)
        if metadata_dir is None:
            searched = ", ".join(str(path) for path in _candidate_raw_dirs(raw_dir))
            raise FileNotFoundError(f"SMPD train metadata not found under: {searched}")
        print(f"Reading SMPD metadata from {metadata_dir}")
        read_json = lambda file_name: _read_json_from_dir(metadata_dir, file_name)
        read_lines = lambda file_name: _read_lines_from_dir(metadata_dir, file_name)

    return {
        "additional": read_json(TRAIN_FILES["additional"]),
        "category": read_json(TRAIN_FILES["category"]),
        "temporal": read_json(TRAIN_FILES["temporal"]),
        "user": read_json(TRAIN_FILES["user"]),
        "text": read_json(TRAIN_FILES["text"]),
        "label": pd.Series([float(value) for value in read_lines(TRAIN_FILES["label"])]),
        "image_path": pd.Series([_normalize_path(value) for value in read_lines(TRAIN_FILES["image_path"])]),
    }


def encode_tags(word_list):
    word_dict = {}
    encoded_list = []

    for value in word_list:
        key = "" if pd.isna(value) else value
        if key not in word_dict:
            word_dict[key] = len(word_dict) + 1
        encoded_list.append([word_dict[key]])

    return encoded_list


def _row_keys(dataframe):
    return list(zip(dataframe["Uid"].astype(str), dataframe["Pid"].astype(str)))


def _path_key(path):
    parts = _normalize_path(path).split("/")
    if len(parts) < 2:
        return None
    pid = re.sub(r"\.[^.]+$", "", parts[-1])
    return parts[-2], pid


def _validate_metadata(metadata):
    additional = metadata["additional"]
    expected_len = len(additional)

    for name, value in metadata.items():
        if len(value) != expected_len:
            raise ValueError(
                f"SMPD metadata row mismatch: additional has {expected_len} rows, "
                f"but {name} has {len(value)} rows"
            )

    expected_keys = _row_keys(additional)
    for name in ["category", "temporal", "user", "text"]:
        keys = _row_keys(metadata[name])
        mismatch = next((idx for idx, key in enumerate(keys) if key != expected_keys[idx]), None)
        if mismatch is not None:
            raise ValueError(
                f"SMPD {name} metadata is not aligned at row {mismatch}: "
                f"expected {expected_keys[mismatch]}, got {keys[mismatch]}"
            )

    path_keys = [_path_key(path) for path in metadata["image_path"]]
    mismatch = next((idx for idx, key in enumerate(path_keys) if key != expected_keys[idx]), None)
    if mismatch is not None:
        raise ValueError(
            f"SMPD image path is not aligned at row {mismatch}: "
            f"expected {expected_keys[mismatch]}, got {metadata['image_path'].iloc[mismatch]}"
        )

    image_ids = [f"{uid}_{pid}" for uid, pid in expected_keys]
    duplicated = pd.Index(image_ids).duplicated()
    if duplicated.any():
        first_duplicate = image_ids[int(duplicated.argmax())]
        raise ValueError(f"Duplicated SMPD image_id found: {first_duplicate}")


def _safe_text(series):
    return series.fillna("").astype(str)


def _safe_tags(series):
    return _safe_text(series).apply(lambda tags: tags.split()).tolist()


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _timezone_id(user_data):
    if "timezone_id" in user_data.columns:
        return _safe_text(user_data["timezone_id"])
    if "timezone_timezone_id" in user_data.columns:
        return _safe_text(user_data["timezone_timezone_id"])
    return pd.Series([""] * len(user_data))


def process_meta_data(
    train_text,
    train_additional_information,
    train_category,
    train_temporalspatial_information,
    train_user_data,
    label_data,
    path,
    train_img_paths=None,
):
    image_ids = [
        f"{uid}_{pid}"
        for uid, pid in zip(
            train_additional_information["Uid"].astype(str),
            train_additional_information["Pid"].astype(str),
        )
    ]
    if train_img_paths is None:
        train_img_paths = [
            f"train/{uid}/{pid}.jpg"
            for uid, pid in zip(
                train_additional_information["Uid"].astype(str),
                train_additional_information["Pid"].astype(str),
            )
        ]
    label_series = label_data.iloc[:, 0] if isinstance(label_data, pd.DataFrame) else label_data

    dataset = {
        "image_id": image_ids,
        "image_rel_path": [_normalize_path(path_value) for path_value in train_img_paths],
        "text": _safe_text(train_text["Title"]).tolist(),
        "tags": _safe_tags(train_text["Alltags"]),
        "label": pd.to_numeric(label_series, errors="coerce").fillna(0).tolist(),
        "user_id": encode_tags(train_additional_information["Uid"].astype(str)),
        "pathalias": encode_tags(_safe_text(train_additional_information["Pathalias"])),
        "category": encode_tags(_safe_text(train_category["Category"])),
        "subcategory": encode_tags(_safe_text(train_category["Subcategory"])),
        "concepts": encode_tags(_safe_text(train_category["Concept"])),
        "postdate": train_temporalspatial_information["Postdate"].fillna(0).tolist(),
        "photo_firstdate": _safe_text(train_user_data["photo_firstdate"]).tolist(),
        "photo_firstdatetaken": _safe_text(train_user_data["photo_firstdatetaken"]).tolist(),
        "photo_count": _safe_numeric(train_user_data["photo_count"]).tolist(),
        "time_zone_id": _timezone_id(train_user_data).tolist(),
        "time_zone_offset": _safe_text(train_user_data["timezone_offset"]).tolist(),
    }

    dataframe = pd.DataFrame(dataset)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_pickle(output_path)
    return dataframe


def build_dataset(raw_dir=DEFAULT_RAW_DIR, metadata_zip=None, output_path="datasets/SMPD/dataset.pkl"):
    metadata = _read_metadata(raw_dir, metadata_zip)
    _validate_metadata(metadata)

    dataframe = process_meta_data(
        metadata["text"],
        metadata["additional"],
        metadata["category"],
        metadata["temporal"],
        metadata["user"],
        metadata["label"],
        output_path,
        metadata["image_path"],
    )
    print(
        "SMPD dataset processed: "
        f"{len(dataframe)} UGCs, {dataframe['image_id'].nunique()} unique image ids, "
        f"{len({item[0] for item in _row_keys(metadata['additional'])})} users"
    )
    return dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Build the SMPD dataset pickle from official train metadata.")
    parser.add_argument("--raw_dir", default=str(DEFAULT_RAW_DIR), help="Directory containing SMPD metadata files or zips")
    parser.add_argument("--metadata_zip", default=None, help="Optional path to SMPD train metadata zip")
    parser.add_argument("--output_path", default="datasets/SMPD/dataset.pkl", help="Output dataset pickle path")
    return parser.parse_args()


def main():
    args = parse_args()
    build_dataset(args.raw_dir, args.metadata_zip, args.output_path)


if __name__ == "__main__":
    main()
