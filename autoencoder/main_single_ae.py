import numpy as np
import os
import sys
import time
import argparse

import torch
from torch import nn
import importlib

#from dataset_ae import Clima_dataset, custom_collate_fn

import dataset_ae as dataset
import models_ae as models
import utils_ae as utils

#from utils import train_epoch_multigpu_CNN_GNN as train_epoch
#from utils import train_model_multigpu as train
from utils_ae import load_encoder_checkpoint, load_model_checkpoint
from utils_ae import test_model_ae as test
from utils_ae import check_freezed_layers
from utils_ae import tweedie_loss

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='root path to input dataset')
parser.add_argument('--log_path', type=str, default=os.getcwd(), help='log saving path')

#-- input files
parser.add_argument('--input_file', type=str, default="input_standard.pkl")
#parser.add_argument('--gnn_data_file', type=str, default="gnn_data_standard.pkl")
parser.add_argument('--target_file', type=str, default="gnn_target_2015-2016.pkl")
parser.add_argument('--idx_file', type=str, default="idx_to_key_2015-2016.pkl")
parser.add_argument('--checkpoint_encoder_file', type=str, default="checkpoint_ae.pth")
parser.add_argument('--checkpoint_input_file', type=str, default="checkpoint_input.pth")

#-- output files
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')
parser.add_argument('--checkpoint_file', type=str, default="checkpoint.pth")
parser.add_argument('--loss_file', type=str, default="loss.csv")

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
parser.add_argument('--no-load_ae_checkpoint', dest='load_checkpoint', action='store_false')
parser.add_argument('--test_model',  action='store_true')
parser.add_argument('--no-test_model', dest='test_model', action='store_false')

#--other
parser.add_argument('--model_name', type=str)
parser.add_argument('--train_fn', type=str, default="train_model")
parser.add_argument('--epoch_fn', type=str, default="train_epoch_ae")
parser.add_argument('--test_fn', type=str, default="test_model_ae")
parser.add_argument('--checkpoint_ctd', type=str, default='../checkpoint.pth', help='checkpoint to load to continue')
parser.add_argument('--ctd_training',  action='store_true')
parser.add_argument('--no-ctd_training', dest='ctd_training', action='store_false')

parser.add_argument('--dataset_name', type=str, default="Clima_dataset")
parser.add_argument('--collate_fn_name', type=str, default="custom_collate_fn")

if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    args = parser.parse_args()

    Model = getattr(models, args.model_name)
    Dataset = getattr(dataset, args.dataset_name)
    custom_collate_fn = getattr(dataset, args.collate_fn_name)
    train = getattr(utils, args.train_fn)
    train_epoch = getattr(utils, args.epoch_fn)
    test = getattr(utils, args.test_fn)

    with open(args.log_path+args.log_file, 'w') as f:
        f.write(f'\nStarting on with pct_trainset={args.pct_trainset}, lr={args.lr} and epochs={args.epochs}.'+
                f'\nThere are {torch.cuda.device_count()} available GPUs.')
        
    #-- create the dataset
    dataset = Dataset(path=args.input_path, input_file=args.input_file, idx_file=args.idx_file, target_file=args.target_file)

    #-- split into trainset and testset
    len_trainset = int(len(dataset) * args.pct_trainset)
    len_testset = len(dataset) - len_trainset
    
    with open(args.log_path+args.log_file, 'a') as f:
        f.write(f'\nTrainset size = {len_trainset}, testset size = {len_testset}.')

    generator=torch.Generator().manual_seed(42)
    trainset, testset = torch.utils.data.random_split(dataset, lengths=(len_trainset, len_testset), generator=generator)

    #-- construct the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    with open(args.log_path+args.log_file, 'a') as f:
        f.write(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %")
    
    #-- define the model
    model = Model()

    #-- either load the model checkpoint or load the parameters for the encoder
    if args.load_ae_checkpoint is True and args.checkpoint_ctd is False:
        model = load_encoder_checkpoint(model, args.checkpoint_encoder_file, args.log_path, args.log_file, fine_tuning=args.fine_tuning)
    elif args.load_ae_checkpoint is True and args.checkpoint_ctd is True:
        raise RuntimeError("Either load the ae parameters or continue the training.")

    #-- train the model
    #loss_fn = nn.functional.mse_loss
    loss_fn = nn.functional.l1_loss
    if args.load_ae_checkpoint and args.fine_tuning:
        optimizer =  torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if 'encoder' not in name], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=.1)

    model.cuda()
    #model, optimizer, trainloader, testloader = accelerator.prepare(model, optimizer, trainloader, testloader)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    with open(args.log_path+args.log_file, 'a') as f:
        f.write(f"\nTotal number of trainable parameters: {total_params}.")

    check_freezed_layers(model, args.log_path, args.log_file)

    start = time.time()

    total_loss, loss_list = train(model=model, dataloader=trainloader, loss_fn=loss_fn, optimizer=optimizer,
        num_epochs=args.epochs, log_path=args.log_path, log_file=args.log_file, lr_scheduler=scheduler,
        checkpoint_name=args.log_path+args.checkpoint_file, loss_name=args.log_path+args.loss_file, train_epoch=train_epoch,
        ctd_training=args.ctd_training, checkpoint_ctd=args.checkpoint_ctd)

    end = time.time()

    with open(args.log_path+args.log_file, 'a') as f:
        f.write(f"\nTraining _cnn_gnn completed in {end - start} seconds.")

    #-- test the model
    #if test_model:
    test_loss_total, test_loss_avg = test(model, testloader, args.log_path, args.log_file, loss_fn=loss_fn)
    print(f"\nDONE! :) with test loss = {test_loss_avg}")
    
