import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_utils import run_retrieval_pipeline


SCALAR_FEATURES = ["comment_num", "user_id", "taken_timestamp"]
LIST_FEATURES = []


def parse_args():
    parser = argparse.ArgumentParser(description="Split Instagram and retrieve top-K train-pool neighbors.")
    parser.add_argument("--dataset_path", default="datasets/INS/dataset.pkl", help="Input Instagram dataset pickle")
    parser.add_argument("--output_dir", default=None, help="Output directory for train/valid/test/retrieval_pool")
    parser.add_argument("--retrieval_num", default=50, type=int, help="Number of retrieved UGCs per query")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for train/valid/test split")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    run_retrieval_pipeline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        retrieval_num=args.retrieval_num,
        scalar_features=SCALAR_FEATURES,
        list_features=LIST_FEATURES,
        seed=args.seed,
        dataset_name="INS",
        weight_mode="idf",
        tie_break="desc",
        include_zero_score_candidates=True,
    )
    print(f"Runtime: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
