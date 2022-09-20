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

    target_classes = dict()
    target_regression = dict()
    idx_to_key_not_all_zero = []

    for k, vals in target.items():
        vals_classes = np.where(vals >= threshold, 1, 0)
        target_classes[k] = vals_classes.astype(np.float32)
        vals_regression = np.where(vals >= threshold, np.log(vals), 0)
        target_regression[k] = vals_regression.astype(np.float32)
        if not (vals_classes == 0).all():
            idx_to_key_not_all_zero.append(k)

    idx_to_key_not_all_zero = np.array(idx_to_key_not_all_zero)

    # write the files
    with open(args.log_path+args.log_file, 'a') as f:
        f.write("\nWriting the files...")

    with open(args.output_path+args.out_classes_file, 'wb') as f:
        pickle.dump(target_classes, f)

    with open(args.log_path+args.log_file, 'a') as f:
        f.write("\nClasses written!")

    with open(args.output_path+args.out_regression_file, 'wb') as f:
        pickle.dump(target_regression, f)

    with open(args.log_path+args.log_file, 'a') as f:
        f.write("\nRegression written!")

    with open(args.output_path+"idx_to_key_not_all_zero_north.pkl", 'wb') as f:
        pickle.dump(idx_to_key_not_all_zero, f)

    with open(args.log_path+args.log_file, 'a') as f:
        f.write("\nDone!")

