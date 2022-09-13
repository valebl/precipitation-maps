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

from utils import load_encoder_checkpoint, check_freezed_layers
from utils import train_model
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
parser.add_argument('--idx_file', type=str)
parser.add_argument('--checkpoint_ae_file', type=str)

#-- output files
parser.add_argument('--out_log_file', type=str, default='log.txt', help='log file')
parser.add_argument('--out_checkpoint_file', type=str, default="checkpoint.pth")
parser.add_argument('--out_loss_file', type=str, default="loss.csv")

#-- training hyperparameters
parser.add_argument('--pct_trainset', type=float, default=0.8, help='percentage of dataset in trainset')
parser.add_argument('--epochs', type=int, default=15, help='number of total training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size (global)')
parser.add_argument('--step_size', type=int, default=10, help='scheduler step size (global)')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (wd)')
parser.add_argument('--fine_tuning',  action='store_true')
parser.add_argument('--no-fine_tuning', dest='fine_tuning', action='store_false')
parser.add_argument('--load_ae_checkpoint',  action='store_true')
parser.add_argument('--no-load_ae_checkpoint', dest='load_ae_checkpoint', action='store_false')

#-- boolean
parser.add_argument('--checkpoint_ctd', type=str, help='checkpoint to load to continue')
parser.add_argument('--ctd_training',  action='store_true')
parser.add_argument('--no-ctd_training', dest='ctd_training', action='store_false')
parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')
parser.add_argument('--test_model',  action='store_true')
parser.add_argument('--no-test_model', dest='test_model', action='store_false')

#-- other
parser.add_argument('--model_name', type=str)
parser.add_argument('--loss_fn', type=str, default="mse_loss")
parser.add_argument('--net_type', type=str)
#parser.add_argument('--train_fn', type=str, default="train_model_multigpu")
#parser.add_argument('--epoch_fn', type=str, default="train_epoch_multigpu_CNN_GNN")
#parser.add_argument('--test_fn', type=str, default="test_model")
#parser.add_argument('--dataset_name', type=str, default="Clima_dataset")
#parser.add_argument('--collate_fn_name', type=str, default="custom_collate_fn")
parser.add_argument('--performance', type=str, default=None)

if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    Model = getattr(models, args.model_name)
    #loss_fn = getattr(nn.functional, args.loss_fn)
    train_epoch = getattr(utils, "train_epoch_"+args.net_type)
    test_model = getattr(utils, "test_model_"+args.net_type)
    custom_collate_fn = getattr(dataset, "custom_collate_fn_"+args.net_type)

    if args.loss_fn == 'weighted_mse_loss' or args.loss_fn == 'mse_loss_mod':
        loss_fn = getattr(utils, args.loss_fn)
    else:
        loss_fn = getattr(nn.functional, args.loss_fn)

    if args.use_accelerate is True:
        accelerator = Accelerator()
    else:
        accelerator = None

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.out_log_file, 'w') as f:
            f.write(f'Cuda is available: {torch.cuda.is_available()}.\nStarting with pct_trainset={args.pct_trainset}, lr={args.lr} and epochs={args.epochs}.'+
                f'\nThere are {torch.cuda.device_count()} available GPUs.'+
                f'\nModel = {args.model_name}, batch size = {args.batch_size}')
        
    #-- create the dataset
    dataset = Dataset(path=args.input_path, input_file=args.input_file, data_file=args.data_file,
        target_file=args.target_file, idx_file=args.idx_file, net_type=args.net_type)

    #-- split into trainset and testset
    generator=torch.Generator().manual_seed(42)
    len_trainset = int(len(dataset) * args.pct_trainset)
    len_testset = len(dataset) - len_trainset
    trainset, testset = torch.utils.data.random_split(dataset, lengths=(len_trainset, len_testset), generator=generator)
    
    # split trainset into trainset and validationset
    len_trainset = int(len(trainset) * 0.8)
    len_validationset = len(trainset) - len_trainset
    trainset, validationset = torch.utils.data.random_split(trainset, lengths=(len_trainset, len_validationset), generator=generator)

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.out_log_file, 'a') as f:
            f.write(f'\nTrainset size = {len_trainset}, testset size = {len_testset}, validationset size = {len_validationset}.')

    #-- construct the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        with open(args.output_path+args.out_log_file, 'a') as f:
            f.write(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %")
    
    model = Model()

    #-- either load the model checkpoint or load the parameters for the encoder
    if args.load_ae_checkpoint is True and args.ctd_training is False:
        model = load_encoder_checkpoint(model, args.checkpoint_ae_file, args.output_path, args.out_log_file, accelerator,
                fine_tuning=args.fine_tuning, net_name="dwn_conv")
    elif args.load_ae_checkpoint is True and args.ctd_training is True:
        raise RuntimeError("Either load the ae parameters or continue the training.")

    #-- define the optimizer and trainable parameters
    if args.load_ae_checkpoint and args.fine_tuning:
        optimizer =  torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=.9)
    else:
        optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if 'encoder' not in name], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=0.1, cycle_momentum=False)

    if accelerator is not None:
        model, optimizer, trainloader, testloader = accelerator.prepare(model, optimizer, trainloader, testloader)
    else:
        model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator is None or accelerator.is_main_process: 
        with open(args.output_path+args.out_log_file, 'a') as f:
            f.write(f"\nTotal number of trainable parameters: {total_params}.")

    check_freezed_layers(model, args.output_path, args.out_log_file, accelerator)

    start = time.time()

    total_loss, loss_list = train_model(model=model, dataloader=trainloader, loss_fn=loss_fn, optimizer=optimizer,
        num_epochs=args.epochs, accelerator=accelerator, log_path=args.output_path, log_file=args.out_log_file, lr_scheduler=scheduler,
        checkpoint_name=args.output_path+args.out_checkpoint_file, loss_name=args.output_path+args.out_loss_file, train_epoch=train_epoch,
        ctd_training=args.ctd_training, checkpoint_ctd=args.checkpoint_ctd, performance=args.performance, validationloader=validationloader)

    end = time.time()

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.out_log_file, 'a') as f:
            f.write(f"\nTraining _cnn_gnn completed in {end - start} seconds.")

    #-- test the model
    if args.test_model:
        test_loss_total, test_loss_avg = test_model(model, trainloader, args.output_path, args.out_log_file, accelerator, loss_fn=nn.functional.mse_loss)  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.out_log_file, 'a') as f:
            f.write(f"\nDONE! :)")

