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
parser.add_argument('--out_classes_file', type=str)
parser.add_argument('--out_mean_file', type=str)
parser.add_argument('--log_file', type=str)

if __name__ == '__main__':
    
    args = parser.parse_args()

    with open(args.log_path+args.log_file, 'w') as f:
        f.write("\nStarting...")

    with open(args.input_path+args.target_file, 'rb') as f:
        target = pickle.load(f)

    target_classes = dict()
    target_mean = dict()
    
    for k, vals in target.items():
        vals_classes = np.where(vals.reshape(vals.shape[0]) > 0)
        vals_mean = np.mean(vals)
        target_classes[k] = vals_classes
        target_mean[k] = vals_mean

    with open(args.log_path+args.log_file, 'a') as f:
        f.write("\nWritign the files...")

    with open(args.output_path+args.out_classes_file, 'wb') as f:
        pickle.dump(target_classes, f)

    with open(args.output_path+args.out_mean_file, 'wb') as f:
        pickle.dump(target_mean, f)

    with open(args.log_path+args.log_file, 'a') as f:
        f.write("\nDone!")

