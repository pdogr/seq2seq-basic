import argparse
import csv
import os
import random
from string import ascii_letters, digits, punctuation

from utils import ensure_dir

def generate_dataset(
        root, name,
        size, max_len, vocab_size):
    valid_letters = []
    if vocab_size == 'small':
        valid_letters = digits
    elif vocab_size == 'medium':
        valid_letters = ascii_letters+digits
    elif vocab_size == 'large':
        valid_letters = ascii_letters+digits+punctuation

    path = os.path.join(root, vocab_size+str(max_len), name)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    data_path = os.path.join(path, name+'_data.txt')

    print("Generate {} of vocab {}".format(name, len(valid_letters)))
    with open(data_path, 'w') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerow(['source', 'target'])
        for _ in range(size):
            length = random.randint(1, max_len)
            seq = []
            for _ in range(length):
                seq.append(random.choice(valid_letters))
            target = list(reversed(seq))
            writer.writerow([" ".join(seq), " ".join(target)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='data directory', required=True)
    parser.add_argument('--max-len', help='max seq length', required=True,
                        type=int)
    parser.add_argument('--vocab-size', help="size of vocab", required=True,
                        choices=['small', 'medium', 'large'])
    args = parser.parse_args()
    root_dir = args.dir
    dataset_dir = os.path.join(root_dir, 'reverse')
    ensure_dir(dataset_dir)
    generate_dataset(
        dataset_dir, 'train',
        30000, args.max_len, args.vocab_size)
    generate_dataset(
        dataset_dir, 'test',
        8000, args.max_len, args.vocab_size)
