import numpy as np
import os
import sys
import time
import argparse

import torch
from torch import nn

from dataset import Clima_dataset, custom_collate_fn
from models import CNN_GNN_deep_3 as Model
from utils import train_epoch_multigpu_CNN_GNN as train_epoch
from utils import train_model_multigpu_CNN_GNN as train
from utils import load_encoder_checkpoint, load_model_checkpoint, test_model

from accelerate import Accelerator

accelerator = Accelerator()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='root path to input dataset')
parser.add_argument('--log_path', type=str, default=os.getcwd(), help='log saving path')

#-- input files
parser.add_argument('--input_file', type=str, default="input_standard.pkl")
parser.add_argument('--gnn_data_file', type=str, default="gnn_data_standard.pkl")
parser.add_argument('--gnn_target_file', type=str, default="gnn_target_2015-2016.pkl")
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
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (wd)')
parser.add_argument('--fine_tuning',  action='store_true')
parser.add_argument('--no-fine_tuning', dest='fine_tuning', action='store_false')
parser.add_argument('--load_checkpoint',  action='store_true')
parser.add_argument('--no-load_checkpoint', dest='load_checkpoint', action='store_false')


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    args = parser.parse_args()

    with open(log_path+log_file, 'w') as f:
        f.write(f'\nStarting on with pct_trainset={args.pct_trainset}, lr={args.lr} and epochs={args.epochs}.'+
                f'\nThere are {torch.cuda.device_count()} available GPUs.')
        
    #-- create the dataset
    dataset = Clima_dataset(path=args.input_path, input_file=args.input_file, data_file=args.gnn_data_file, target_file=args.gnn_target_file, idx_file=args.idx_file)

    #-- split into trainset and testset
    len_trainset = int(len(dataset) * args.pct_trainset)
    len_testset = len(dataset) - len_trainset
    
    if accelerator.is_main_process:
        with open(args.log_path+log_file, 'a') as f:
            f.write(f'\nTrainset size = {len_trainset}, testset size = {len_testset}.')

    generator=torch.Generator().manual_seed(42)
    trainset, testset = torch.utils.data.random_split(dataset, lengths=(len_trainset, len_testset), generator=generator)

    #-- construct the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    if accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        with open(args.log_path+log_file, 'a') as f:
            f.write(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %")
    
    #-- define the model
    model = Model(input_size=25)

    #-- either load the model checkpoint or load the parameters for the encoder
    if args.load_checkpoint is True:
        model = load_model_checkpoint(model, args.checkpoint_input_file, accelerator, args.log_path, log_file)
    else:
        model = load_encoder_checkpoint(model, args.checkpoint_encoder_file, accelerator, args.log_path, log_file, fine_tuning=args.fine_tuning)

    #-- train the model
    loss_fn = nn.functional.mse_loss
    if args.fine_tuning:
        optimizer =  torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if 'encoder' not in name], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)

    model, optimizer, trainloader = accelerator.prepare(model, optimizer, trainloader)

    start = time.time()

    total_loss, loss_list = train(model=model, dataloader=trainloader, loss_fn=loss_fn, optimizer=optimizer,
        num_epochs=epochs, accelerator=accelerator, log_path=args.log_path, log_file=log_file, lr_scheduler=scheduler,
        checkpoint_name=args.log_path+args.checkpoint_file, loss_name=args.log_path+args.loss_file, train_epoch=train_epoch)

    end = time.time()

    if accelerator.is_main_process:
        with open(args.log_path+log_file, 'a') as f:
            f.write(f"Training _cnn_gnncompleted in {start - end} seconds.")

    #-- test the model
    test_loss = test_model(model, testloader, accelerator, loss_fn=None)


