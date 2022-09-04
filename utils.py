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
        self.avg_iter_list = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def add_loss(self):
        self.avg_list.append(self.avg)
    
    def add_iter_loss(self):
        self.avg_iter_list.append(self.avg)


def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def accuracy(nn_output, ground_truth, k=1):
    '''
    Return accuracy@k for the given model output and ground truth
    nn_output: a tensor of shape (num_datapoints x num_classes) which may 
       or may not be the output of a softmax or logsoftmax layer
    ground_truth: a tensor of longs or ints of shape (num_datapoints)
    k: the 'k' in accuracy@k
    '''
    assert k <= nn_output.shape[1], f"k too big. Found: {k}. Max: {nn_output.shape[1]} inferred from the nn_output"
    # get classes of assignment for the top-k nn_outputs row-wise
    nn_out_classes = nn_output.topk(k).indices
    # make ground_truth a column vector
    ground_truth_vec = ground_truth.unsqueeze(-1)
    # and repeat the column k times (= reproduce nn_out_classes shape)
    ground_truth_vec = ground_truth_vec.expand_as(nn_out_classes)
    # produce tensor of booleans - at which position of the nn output is the correct class located?
    correct_items = (nn_out_classes == ground_truth_vec)
    # now getting the accuracy is easy, we just operate the sum of the tensor and divide it by the number of examples
    acc = correct_items.sum().item() / nn_output.shape[0]
    return acc


def load_encoder_checkpoint(model, checkpoint, log_path, log_file, accelerator, fine_tuning=True, net_name='encoder'):
    with open(log_path+log_file, 'a') as f:
        f.write("\nLoading encoder parameters.") 
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint["parameters"]
    for name, param in state_dict.items():
        #with open(log_path+log_file, 'a') as f:
        #   f.write(f"\n{name}, {net_name in name}")
        if net_name in name:
            if accelerator is None or accelerator.is_main_process:
                with open(log_path+log_file, 'a') as f:
                    f.write(f"\nLoading parameters '{name}'")
            param = param.data
            #layer = name.partition("module.")[2]
            model.state_dict()[name].copy_(param)
    if not fine_tuning:
        [param.requires_grad_(False) for name, param in model.named_parameters() if net_name in name]
    return model


def load_model_checkpoint(model, checkpoint, log_path, log_file, accelerator):
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint["parameters"]
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nLoading checkpoint")
    for name, param in state_dict.items():
        param = param.data
        #layer = name.partition("module.")[2]
        model.state_dict()[name].copy(param)
    return model


def check_freezed_layers(model, log_path, log_file, accelerator):
    for name, param in model.named_parameters():
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nLayer {name} requires_grad = {param.requires_grad}")


#------Training utilities------

#------EPOCH LOOPS------  

def train_epoch_ae(model, dataloader, loss_fn, optimizer, loss_meter, accelerator):
    
    for X in dataloader:
        if accelerator is None:
            X = X.cuda()
        optimizer.zero_grad()
        X_pred = model(X)
        loss = loss_fn(X_pred, X)
        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)
        #torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
        loss_meter.update(val=loss.item(), n=X.shape[0])
        loss_meter.add_iter_loss()

def train_epoch_cnn(model, dataloader, loss_fn, optimizer, loss_meter, accelerator):

    for X, y in dataloader:
        if accelerator is None:
            X = X.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        y_pred = model(X).squeeze()
        loss = loss_fn(y_pred, y)
        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)
        #torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
        loss_meter.update(val=loss.item(), n=X.shape[0])
        loss_meter.add_iter_loss()

def train_epoch_gnn(model, dataloader, loss_fn, optimizer, loss_meter, accelerator):

    for X, data in dataloader:
        device = accelerator.device if accelerator is None else 'cuda'
        optimizer.zero_grad()
        y_pred, y = model(X, data, device)
        loss = loss_fn(y_pred, y)
        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)
        #torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
        loss_meter.update(val=loss.item(), n=X.shape[0])    
        loss_meter.add_iter_loss()    

#------ TRAIN ------  

def train_model(model, dataloader, loss_fn, optimizer, num_epochs,
        log_path, log_file, train_epoch, accelerator, lr_scheduler=None, 
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
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.5f}")
        start_time = time.time()
        train_epoch(model, dataloader, loss_fn, optimizer, loss_meter, accelerator)
        end_time = time.time()
        loss_meter.add_loss()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} completed in {end_time - start_time:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.4f}.")
        
        if accelerator is None or accelerator.is_main_process and (epoch+1) % save_interval == 0 and (epoch+1) != num_epochs:
            np.savetxt(loss_name, loss_meter.avg_list)
            checkpoint_dict = {
                "parameters": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                #"loss": loss_meter.avg
                }
            torch.save(checkpoint_dict, checkpoint_name)

    if accelerator is None or accelerator.is_main_process:
        np.savetxt(loss_name, loss_meter.avg_list)
        checkpoint_dict = {
            "parameters": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg             
            }
        torch.save(checkpoint_dict, checkpoint_name)
    
    return loss_meter.sum, loss_meter.avg_list


#------ TEST ------  

def test_model_ae(model, dataloader, log_path, log_file, accelerator, loss_fn=None):
    
    if loss_fn is not None:
        loss_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for X in dataloader:
            if accelerator is None:
                X = X.cuda()
            X_pred = model(X)
            loss = loss_fn(X_pred, X) if loss_fn is not None else None
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
    
    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                    +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}")
    return fin_loss_total, fin_loss_avg

def test_model_cnn(model, dataloader, log_path, log_file, accelerator=None, loss_fn=None):
    if loss_fn is not None:
        loss_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            if accelerator is None:
                X = X.cuda()
                y = y.cuda()
            y_pred = model(X).squeeze()
            loss = loss_fn(y_pred, y) if loss_fn is not None else None
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])

    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                    +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}")
    return fin_loss_total, fin_loss_avg


def test_model_gnn(model, dataloader, accelerator, log_path, log_file, loss_fn=None):
    if loss_fn is not None:
        loss_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for X, data in dataloader:
            device = accelerator.device if accelerator is None else 'cuda'
            y_pred, y = model(X, data, device)
            loss = loss_fn(y_pred, y) if loss_fn is not None else None
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
    
    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                    +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}") 
    return fin_loss_total, fin_loss_avg 
