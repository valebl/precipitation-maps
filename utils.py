import numpy as np
import time

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


def load_encoder_checkpoint(model, checkpoint, accelerator, log_path, log_file, fine_tuning=True):
    state_dict = torch.load(checkpoint)
    for name, param in state_dict.items():
        if 'encoder' in name:
            if accelerator.is_main_process:
                with open(log_path+log_file, 'a') as f:
                    f.write(f"\nLoading parameters '{name}'")
            param = param.data
            layer = name.partition("module.")[2]
            model.state_dict()[layer].copy_(param)
    if not fine_tuning:
        [param.requires_grad_(False) for name, param in model.named_parameters() if 'encoder' in name]
    return model


def load_model_checkpoint(model, checkpoint, accelerator, log_path, log_file):
    state_dict = torch.load(checkpoint)
    with open(log_path+log_file, 'a') as f:
        f.write(f"\nLoading checkpoint")
    for name, param in state_dict.items():
        param = param.data
        layer = name.partition("module.")[2]
        model.state_dict()[layer].copy(param)
    return model


def check_freezed_layers(model, accelerator, log_path, log_file):
    for name, param in model.named_parameters():
        if accelerator.is_main_process:
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
        data = Batch.from_data_list(data)
        X, data = X.cuda(), data.cuda()
        y = data.y
        optimizer.zero_grad()
        y_pred = model(X, data, 'cuda')
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
        data = Batch.from_data_list(data)
        X, data = X.cuda(), data.cuda()
        y = data.y
        optimizer.zero_grad()
        y_pred = model(X, data, 'cuda')
        loss = loss_fn(y_pred, y)
        accelerator.backward(loss)
        optimizer.step()
        loss_meter.update(val=loss.item(), n=X.shape[0])


#------TRAIN ON SINGLE-GPU------  

def train_model(model, dataloader, loss_fn, optimizer, num_epochs,
        log_path, log_file, train_epoch, lr_scheduler=None, checkpoint_name="checkpoint.pth", loss_name="loss.csv"):

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

        if epoch % 5 == 0 and epoch != num_epochs-1:
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
        accelerator, log_path, log_file, train_epoch, lr_scheduler=None, checkpoint_name="checkpoint.pth", loss_name="loss.csv", gnn=True):
    
    model.train()
    # epoch loop
    for epoch in range(num_epochs):
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
        
        if accelerator.is_main_process and epoch % 5 == 0 and epoch != num_epochs-1:
            np.savetxt(loss_name, loss_meter.avg_list)
            checkpoint_dict = {
                "parameters": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
                }
            torch.save(checkpoint_dict, checkpoint_name)

    if accelerator.is_main_process:
        np.savetxt(loss_name, loss_meter.avg_list)
        checkpoint_dict = {
            "parameters": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
            }
        torch.save(checkpoint_dict, checkpoint_name)
    
    return loss_meter.sum, loss_meter.avg_list


#------TEST ON MULTI-GPU------  

def test_model(model, dataloader, accelerator, log_path, log_file, loss_fn=None):
    if loss_fn is not None:
        loss_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for X, data in dataloader:
            y = data.y
            y_pred = model(X, data, accelerator.device)
            loss = loss_fn(y, y_pred) if loss_fn is not None else None
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
    
    fin_loss = loss_meter.sum if loss_fn is not None else None
    if accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss {fin_loss if fin_loss is not None else '--'}") 
    return fin_loss
