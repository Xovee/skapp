import os
import argparse


def disassemble(path, output_path, retrieval_num):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(
        f"Skipped materializing {output_path}. "
        f"Single-item training expands {path} dynamically with {retrieval_num} retrievals."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dynamic SMPD single-item training paths.")
    parser.add_argument("--retrieval_num", default=50, type=int, help="Number of retrieved UGCs per query")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    retrieval_num = args.retrieval_num

    source_path = r'datasets/SMPD'
    disassemble_path = r'datasets/SMPD_dissembled'

    os.makedirs(disassemble_path, exist_ok=True)

    disassemble(os.path.join(source_path, 'train.pkl'), os.path.join(disassemble_path, 'train.pkl'), retrieval_num)
    disassemble(os.path.join(source_path, 'valid.pkl'), os.path.join(disassemble_path, 'valid.pkl'), retrieval_num)
    disassemble(os.path.join(source_path, 'test.pkl'), os.path.join(disassemble_path, 'test.pkl'), retrieval_num)
