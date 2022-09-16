import numpy as np
import os
import sys
import time
import argparse

import torch
from torch import nn
import importlib

import models
import utils
import dataset

from utils import load_model_checkpoint
from dataset import Clima_dataset as Dataset

from accelerate import Accelerator

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')

#-- input files
parser.add_argument('--input_file', type=str, default="input_standard.pkl")
parser.add_argument('--data_file', type=str, default=None)
parser.add_argument('--target_file', type=str, default=None)
parser.add_argument('--idx_file', type=str, default="idx_to_key.pkl")
parser.add_argument('--checkpoint_input_file', type=str, default="checkpoint_input.pth")

#-- output files
parser.add_argument('--out_log_file', type=str, default='log.txt', help='log file')

#-- training hyperparameters
parser.add_argument('--pct_trainset', type=float, default=0.8, help='percentage of dataset in trainset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size (global)')

#-- boolean
parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')

#-- other
parser.add_argument('--model_name', type=str)
parser.add_argument('--loss_fn', type=str, default="mse_loss")
parser.add_argument('--net_type', type=str)
parser.add_argument('--performance', type=str, default=None)

if __name__ == '__main__':
    
    torch.backends.cudnn.benchmark = True
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    Model = getattr(models, args.model_name)
    loss_fn = getattr(nn.functional, args.loss_fn)
    test_model = getattr(utils, "test_model_"+args.net_type)
    custom_collate_fn = getattr(dataset, "custom_collate_fn_"+args.net_type)

    if args.use_accelerate is True:
        accelerator = Accelerator()
    else:
        accelerator = None

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.out_log_file, 'w') as f:
            f.write(f'Cuda is available: {torch.cuda.is_available()}.\nStarting with pct_trainset={args.pct_trainset}.'+
                    f'\nThere are {torch.cuda.device_count()} available GPUs.')

    #-- create the dataset
    dataset = Dataset(path=args.input_path, input_file=args.input_file, data_file=args.data_file,
            target_file=args.target_file, idx_file=args.idx_file, net_type=args.net_type)

    #-- split into trainset and testset
    len_trainset = int(len(dataset) * args.pct_trainset)
    len_testset = len(dataset) - len_trainset

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.out_log_file, 'a') as f:
            f.write(f'\nTrainset size = {len_trainset}, testset size = {len_testset}.')

    generator=torch.Generator().manual_seed(42)
    trainset, testset = torch.utils.data.random_split(dataset, lengths=(len_trainset, len_testset), generator=generator)

    #-- construct the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        with open(args.output_path+args.out_log_file, 'a') as f:
            f.write(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %")

    model = Model()
    checkpoint = torch.load(args.checkpoint_input_file)
    model.load_state_dict(checkpoint["parameters"])
    
    if accelerator is not None:
        model, optimizer, trainloader, testloader = accelerator.prepare(model, optimizer, trainloader, testloader)
    else:
        model = model.cuda()   
    
    print("Starting the test...")

    #-- test the model
    start = time.time()
    test_loss_total, test_loss_avg = test_model(model, testloader, args.output_path, args.out_log_file, accelerator, loss_fn=loss_fn, performance=args.performance)
    end = time.time()

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.out_log_file, 'a') as f:
            f.write(f"Testing took {end - start}s. Total loss = {test_loss_total}, avg loss = {test_loss_avg}.")
            f.write(f"\nDONE! :)")

    print("Done!")