import os
import time
import argparse


def disassemble(path, output_path, retrieval_num):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(
        f"Skipped materializing {output_path}. "
        f"Single-item training expands {path} dynamically with {retrieval_num} retrievals."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dynamic ICIP single-item training paths.")
    parser.add_argument("--retrieval_num", default=50, type=int, help="Number of retrieved UGCs per query")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    retrieval_num = args.retrieval_num
    source_path = r'datasets/ICIP'
    disassemble_path = r'datasets/ICIP_dissembled'

    os.makedirs(disassemble_path, exist_ok=True)

    disassemble(os.path.join(source_path, 'train.pkl'), os.path.join(disassemble_path, 'train.pkl'), retrieval_num)
    print("[1] Dynamic train disassembly ready.")
    disassemble(os.path.join(source_path, 'valid.pkl'), os.path.join(disassemble_path, 'valid.pkl'), retrieval_num)
    print("[2] Dynamic valid disassembly ready.")
    disassemble(os.path.join(source_path, 'test.pkl'), os.path.join(disassemble_path, 'test.pkl'), retrieval_num)
    print("[3] Dynamic test disassembly ready.")

    print(f"Runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
