import os


def disassemble(path, output_path, retrieval_num):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(
        f"Skipped materializing {output_path}. "
        f"Single-item training expands {path} dynamically with {retrieval_num} retrievals."
    )


if __name__ == "__main__":
    retrieval_num = 200

    source_path = r'datasets/origin'
    disassemble_path = r'datasets/dissembled'

    os.makedirs(disassemble_path, exist_ok=True)

    disassemble(os.path.join(source_path, 'train.pkl'), os.path.join(disassemble_path, 'train.pkl'), retrieval_num)
    disassemble(os.path.join(source_path, 'valid.pkl'), os.path.join(disassemble_path, 'valid.pkl'), retrieval_num)
    disassemble(os.path.join(source_path, 'test.pkl'), os.path.join(disassemble_path, 'test.pkl'), retrieval_num)
