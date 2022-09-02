import numpy as np
import time
import sys

import torch
from torch_geometric.data import Batch

#------Some useful utilities------

class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.avg_list = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def add_loss(self):
        self.avg_list.append(self.avg)


def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_encoder_checkpoint(model, checkpoint, log_path, log_file, fine_tuning=True, accelerator=None):
    state_dict = torch.load(checkpoint)
    for name, param in state_dict.items():
        if 'encoder' in name:
            if accelerator is None or accelerator.is_main_process:
                with open(log_path+log_file, 'a') as f:
                    f.write(f"\nLoading parameters '{name}'")
            param = param.data
            layer = name.partition("module.")[2]
            model.state_dict()[layer].copy_(param)
    if not fine_tuning:
        [param.requires_grad_(False) for name, param in model.named_parameters() if 'encoder' in name]
    return model


def load_model_checkpoint(model, checkpoint, log_path, log_file, accelerator=None):
    state_dict = torch.load(checkpoint)
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nLoading checkpoint")
    for name, param in state_dict.items():
        param = param.data
        layer = name.partition("module.")[2]
        model.state_dict()[layer].copy(param)
    return model

def check_freezed_layers(model, log_path, log_file, accelerator=None):
    for name, param in model.named_parameters():
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nLayer {name} requires_grad = {param.requires_grad}")


#------Training utilities------

#------EPOCH LOOPS------  

def train_epoch_ae(model, dataloader, loss_fn, optimizer,
        loss_meter, accelerator):
    
    for X, _ in dataloader:
        optimizer.zero_grad()
        target = model(X)
        loss = loss_fn(X, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
        loss_meter.update(val=loss.item(), n=X.shape[0])

def train_epoch_CNN_GNN(model, dataloader, loss_fn, optimizer, loss_meter):

    for X, data in dataloader:
        optimizer.zero_grad()
        y_pred, y = model(X, data, 'cuda')
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        loss_meter.update(val=loss.item(), n=X.shape[0])

def train_epoch_ae_multigpu(model, dataloader, loss_fn, optimizer,
        loss_meter, accelerator):
    
    for X, _ in dataloader:
        optimizer.zero_grad()
        target = model(X)
        loss = loss_fn(X, target)
        accelerator.backward(loss)
        #torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
        loss_meter.update(val=loss.item(), n=X.shape[0])


def train_epoch_multigpu_CNN_GNN(model, dataloader, loss_fn, optimizer, 
        loss_meter, accelerator):

    for X, data in dataloader:
        optimizer.zero_grad()
        y_pred, y = model(X, data, accelerator.device) #, 'cuda')
        loss = loss_fn(y_pred, y)
        accelerator.backward(loss)
        optimizer.step()
        loss_meter.update(val=loss.item(), n=X.shape[0])
        

#------TRAIN ON SINGLE-GPU------  

def train_model(model, dataloader, loss_fn, optimizer, num_epochs,
        log_path, log_file, train_epoch, lr_scheduler=None,
        checkpoint_name="checkpoint.pth", loss_name="loss.csv",
        save_interval=1):

    model.train()
    # epoch loop
    for epoch in range(num_epochs):
        loss_meter = AverageMeter()
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.5f}")
        start_time = time.time()
        train_epoch(model, dataloader, loss_fn, optimizer, loss_meter)
        end_time = time.time()
        loss_meter.add_loss()
        if lr_scheduler is not None:
            lr_scheduler.step()
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nEpoch {epoch+1} completed in {end_time - start_time:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.4f}.")

        if (epoch+1) % save_interval == 0 and (epoch+1) != num_epochs:
            np.savetxt('loss.csv', loss_meter.avg_list)
            checkpoint_dict = {
                "parameters": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
                }
            torch.save(checkpoint_dict, checkpoint_name)
            np.savetxt('loss.csv', loss_meter.avg_list)

    checkpoint_dict = {
            "parameters": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
            }
    torch.save(checkpoint_dict, checkpoint_name)
    np.savetxt('loss.csv', loss_meter.avg_list)
        
    return loss_meter.sum, loss_meter.avg_list


#------TRAIN ON MULTI-GPU------  

def train_model_multigpu(model, dataloader, loss_fn, optimizer, num_epochs,
        accelerator, log_path, log_file, train_epoch, lr_scheduler=None, 
        checkpoint_name="checkpoint.pth", loss_name="loss.csv",
        ctd_training=False, checkpoint_ctd="../checkpoint.pth",
        save_interval=1):
    
    epoch_start = 0

    if ctd_training:
        checkpoint = torch.load(checkpoint_ctd)
        model.load_state_dict(checkpoint["parameters"])
        #optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_start = checkpoint["epoch"] + 1
        #loss = checkpoint["loss"]
    
    model.train()
    # epoch loop
    for epoch in range(epoch_start, epoch_start + num_epochs):
        loss_meter = AverageMeter()
        if accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.5f}")
        start_time = time.time()
        train_epoch(model, dataloader, loss_fn, optimizer, loss_meter, accelerator)
        end_time = time.time()
        loss_meter.add_loss()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} completed in {end_time - start_time:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.4f}.")
        
        if accelerator.is_main_process and (epoch+1) % save_interval == 0 and (epoch+1) != num_epochs:
            np.savetxt(loss_name, loss_meter.avg_list)
            checkpoint_dict = {
                "parameters": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                #"loss": loss_meter.avg
                }
            torch.save(checkpoint_dict, checkpoint_name)

    if accelerator.is_main_process:
        np.savetxt(loss_name, loss_meter.avg_list)
        checkpoint_dict = {
            "parameters": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg             
            }
        torch.save(checkpoint_dict, checkpoint_name)
    
    return loss_meter.sum, loss_meter.avg_list


#------TEST ON MULTI-GPU------  


def test_model_ae(model, dataloader, accelerator, log_path, log_file, loss_fn=None):
    if loss_fn is not None:
        loss_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for X, data in dataloader:
            X_pred = model(X)
            loss = loss_fn(X, X_pred) if loss_fn is not None else None
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
    
    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    if accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                    +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}")
    return fin_loss_total, fin_loss_avg


def test_model(model, dataloader, accelerator, log_path, log_file, loss_fn=None):
    if loss_fn is not None:
        loss_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for X, data in dataloader:
            y_pred, y = model(X, data, accelerator.device)
            loss = loss_fn(y, y_pred) if loss_fn is not None else None
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
    
    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    if accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                    +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}") 
    return fin_loss_total, fin_loss_avg 
