# SKAPP

This is a reference implementation of the SKAPP model described in "Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation", published at AAAI 2025.

## Install

Use Python 3.12 and install PyTorch for your CUDA runtime, then:

```bash
python -m pip install -r requirements.txt
```

## Download data

```bash
python src/download.py --dataset icip
```

Replace `icip` with `smpd` or `instagram` to use another dataset.

Downloads are verified and extracted to `datasets/<dataset>/`. 

Alternatively, download the dataset ZIP from
[Google Drive](https://drive.google.com/drive/folders/1VbRgLHxWpCj_JOKLxdrqd_ZeNzyCZheq)
and extract it into the `datasets/` directory at the project root. For example,
extracting `skapp-icip.zip` should give:

```text
datasets/icip/
├── dataset.json
├── train.npz
├── valid.npz
└── test.npz
```

Original datasets:

- **ICIP**: [Official](http://www.visiongarage.altervista.org/popularitydataset/) | [Our mirror](https://drive.google.com/drive/folders/1eToci8-r0_E-zUuvbqSnAPkmO8aHvm67?usp=sharing)
- **SMPD**: [Official](https://smp-challenge.com/download_image.html)
- **Instagram**: [Official](https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset)

## Train and evaluate

```bash
python src/train.py --dataset icip
python src/evaluate.py --dataset icip
```

Training runs once with seed 12 by default and selects checkpoints by validation MSE.
Use `--seeds` followed by one or more distinct integers to customize the seeds. 
Models are saved to `runs/icip/models/`; evaluation outputs go to
`runs/icip/evaluation/`. Use a fresh output directory for a new run, e.g., `--output-dir runs/icip/experiment2`.

Evaluation verifies saved validation predictions before evaluating the test
split. `result.json` contains each seed's MSE, MAE and Spearman correlation,
plus their mean and sample standard deviation. Predictions include sample IDs
and labels.

The default device is `cuda:0`. Use `--device cpu` for small CPU runs,
`--data-root /path/to/datasets` for another data location, or `--output-dir`
for a custom run directory. Dataset features are loaded onto the selected device.
See `--help` for options.

## Recompute RRCP (optional)

The supplied RRCP was generated with auxiliary teachers trained on the training
split and selected using validation data. To train your own teachers and rebuild
RRCP from the packaged neighbors:

```bash
python src/train_teachers.py --dataset icip --kind all
python src/train_teachers.py --dataset icip --kind single
python src/build_rrcp.py --dataset icip
python src/train.py --dataset-path runs/icip/data-rebuilt --output-dir runs/icip/rebuilt-models
python src/evaluate.py --dataset-path runs/icip/data-rebuilt --training-dir runs/icip/rebuilt-models --output-dir runs/icip/rebuilt-evaluation
```

Rebuilding RRCP creates a separate dataset and preserves the downloaded package.

## Citation

```bibtex
@inproceedings{xu2025improving,
  author = {Xu, Xovee and Zhang, Yifan and Zhou, Fan and Song, Jingkuan},
  title = {Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation},
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
  year = {2025},
  pages = {932--940},
  doi = {10.1609/aaai.v39i1.32078}
}
```

MIT License.
