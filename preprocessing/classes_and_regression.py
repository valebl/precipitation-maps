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
parser.add_argument('--idx_to_key_file', type=str)

#-- output_files
parser.add_argument('--out_classes_file', type=str)
parser.add_argument('--out_regression_file', type=str)
parser.add_argument('--log_file', type=str)


if __name__ == '__main__':
    
    args = parser.parse_args()

    threshold = 0.1 # mm

    with open(args.log_path+args.log_file, 'w') as f:
        f.write("\nStarting...")

    with open(args.input_path+args.target_file, 'rb') as f:
        target = pickle.load(f)

    with open(args.input_path+args.idx_to_key_file, 'rb') as f:
        idx_to_key = pickle.load(f)


    target_classes = dict()
    target_regression = dict()
    

    for k, vals in target.items():
        vals_classes = np.where(vals >= threshold, 1, 0)
        target_classes[k] = vals_classes
        vals_regression = np.where(vals >= threshold, log(vals), 0)
        target_regression[k] = vals_regression


    # write the files
    with open(args.log_path+args.log_file, 'a') as f:
        f.write("\nWriting the files...")

    with open(args.output_path+args.out_classes_file, 'wb') as f:
        pickle.dump(target_classes, f)

    with open(args.output_path+args.out_regression_file, 'wb') as f:
        pickle.dump(target_regression, f)


    with open(args.log_path+args.log_file, 'a') as f:
        f.write("\nDone!")

