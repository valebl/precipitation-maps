import numpy as np
import pickle
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='root path to input dataset')
parser.add_argument('--output_path', type=str)
parser.add_argument('--log_path', type=str, default=os.getcwd(), help='log saving path')

#-- files
parser.add_argument('--target_file', type=str)
parser.add_argument('--out_mean_file', type=str)
parser.add_argument('--log_file', type=str)
parser.add_argument('--idx_to_key_file', type=str)
parser.add_argument('--out_idx_to_key_file', type=str)

if __name__ == '__main__':
    
    args = parser.parse_args()

    with open(args.log_path+args.log_file, 'w') as f:
        f.write("\nStarting...")

    with open(args.input_path+args.target_file, 'rb') as f:
        target = pickle.load(f)

    with open(args.input_path+args.idx_to_key_file, 'rb') as f:
        idx_to_key = pickle.load(f)

    target_mean_non_zero = dict()    
    idx_to_key_non_zero = []

    for k, vals in target.items():
        if vals != 0:
            target_mean_non_zero[k] = vals
    
    for idx, k in enumerate(idx_to_key):
        if k in target_mean_non_zero.keys():
            idx_to_key_non_zero.append(k)

    idx_to_key_non_zero = np.array(idx_to_key_non_zero)

    with open(args.log_path+args.log_file, 'a') as f:
        f.write(f"\nLen idx_to_key_non_zero = {idx_to_key_non_zero.shape[0]}, len target_mean_non_zero = {len(list(target_mean_non_zero.keys()))}.\nWriting the files...")

    with open(args.output_path+args.out_mean_file, 'wb') as f:
        pickle.dump(target_mean_non_zero, f)

    with open(args.output_path+args.out_idx_to_key_file, 'wb') as f:
        pickle.dump(idx_to_key_non_zero, f)

    with open(args.log_path+args.log_file, 'a') as f:
        f.write("\nDone!")

