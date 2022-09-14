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
parser.add_argument('--out_sum_file', type=str)
parser.add_argument('--out_mean_file', type=str)
parser.add_argument('--out_classes_file', type=str)
parser.add_argument('--out_sum_non_zero_file', type=str)
parser.add_argument('--out_mean_non_zero_file', type=str)
parser.add_argument('--out_sum_non_zero_log_file', type=str)
parser.add_argument('--out_mean_non_zero_log_file', type=str)
parser.add_argument('--out_idx_to_key_non_zero_file', type=str)
parser.add_argument('--log_file', type=str)



if __name__ == '__main__':
    
    args = parser.parse_args()

    with open(args.log_path+args.log_file, 'w') as f:
        f.write("\nStarting...")

    with open(args.input_path+args.target_file, 'rb') as f:
        target = pickle.load(f)

    with open(args.input_path+args.idx_to_key_file, 'rb') as f:
        idx_to_key = pickle.load(f)

    target_sum = dict()
    target_mean =dict()
    target_classes = dict()

    target_sum_nz = dict()
    target_sum_nz_log = dict()

    target_mean_nz = dict()
    target_mean_nz_log = dict()

    for k, vals in target.items():
        # sum
        vals_sum = np.sum(vals)
        target_sum[k] = vals_sum
        if vals_sum > 0:
            target_sum_nz[k] = vals_sum
            target_sum_nz_log[k] = np.log(vals_sum)
        # mean
        vals_mean = np.mean(vals)
        target_mean[k] = vals_mean
        if vals_mean > 0:
            target_mean_nz[k] = vals_mean
            target_mean_nz_log[k] = np.log(vals_mean)
        # classes
        vals_classes = np.where(vals > 0, 1, 0)
        target_classes[k] = vals_classes


    idx_to_key_non_zero = []

    for idx, k in enumerate(idx_to_key):
        if k in target_sum_nz.keys():
            idx_to_key_non_zero.append(k)

    idx_to_key_non_zero = np.array(idx_to_key_non_zero)

    # write the files
    with open(args.log_path+args.log_file, 'a') as f:
        f.write("\nWriting the files...")

    with open(args.output_path+args.out_sum_file, 'wb') as f:
        pickle.dump(target_sum, f)

    with open(args.output_path+args.out_sum_non_zero_file, 'wb') as f:
        pickle.dump(target_sum_nz, f)

    with open(args.output_path+args.out_sum_non_zero_log_file, 'wb') as f:
        pickle.dump(target_sum_nz_log, f)

    with open(args.output_path+args.out_mean_file, 'wb') as f:
        pickle.dump(target_mean, f)

    with open(args.output_path+args.out_mean_non_zero_file, 'wb') as f:
        pickle.dump(target_mean_nz, f)

    with open(args.output_path+args.out_mean_non_zero_log_file, 'wb') as f:
        pickle.dump(target_mean_nz_log, f)

    with open(args.output_path+args.out_classes_file, 'wb') as f:
        pickle.dump(target_classes, f)

    with open(args.output_path+args.out_idx_to_key_non_zero_file, 'wb') as f:
        pickle.dump(idx_to_key_non_zero, f)


    with open(args.log_path+args.log_file, 'a') as f:
        f.write("\nDone!")

