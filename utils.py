import numpy as np
import time
import sys
import pickle

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR


#------Some useful utilities------

class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    '''
    def __init__(self):
        self.reset()
        self.avg_list = []
        self.avg_iter_list = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def accuracy(prediction, target):
    #prediction_class = torch.sigmoid(prediction)
    if prediction.shape == target.shape:
        prediction_class = torch.where(prediction > 0.5, 1.0, 0.0) 
    else:
        prediction_class = torch.argmax(prediction, dim=-1).squeeze()
    correct_items = (prediction_class == target)
    acc = correct_items.sum().item() / prediction.shape[0]  
    return acc


def weighted_mse_loss(input_batch, target_batch, weights):
    return (weights * (input_batch - target_batch) ** 2).sum() / weights.sum()

def mse_loss_mod(input_batch, target_batch, alpha=0.25,  device='cuda'):
    return ((input_batch - target_batch) ** 2).sum() / input_batch.shape[0] + alpha * ((torch.log(input_batch+10e-9) - torch.log(target_batch+10e-9)) ** 2).sum() / input_batch.shape[0]

def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def load_encoder_checkpoint(model, checkpoint, log_path, log_file, accelerator, fine_tuning=True, net_names=['encoder', 'gru', 'linear']):
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write("\nLoading encoder parameters.") 
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint["parameters"]
    for name, param in state_dict.items():
        #with open(log_path+log_file, 'a') as f:
        #   f.write(f"\n{name}, {net_name in name}")
        for net_name in net_names:
            if net_name in name:
                if accelerator is None or accelerator.is_main_process:
                    with open(log_path+log_file, 'a') as f:
                        f.write(f"\nLoading parameters '{name}'")
                param = param.data
                if name.startswith("module"):
                    name = name.partition("module.")[2]
                try:
                    model.state_dict()[name].copy_(param)
                except:
                     if accelerator is None or accelerator.is_main_process:
                        with open(log_path+log_file, 'a') as f:
                            f.write(f"\nParam {name} was not loaded..")
    if not fine_tuning:
        for net_name in net_names:
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
        if "module" in name:
            name = name.partition("module.")[2]
        model.state_dict()[name].copy(param)
    return model


def check_freezed_layers(model, log_path, log_file, accelerator):
    for name, param in model.named_parameters():
        n_param = param.numel()
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nLayer {name} requires_grad = {param.requires_grad} and has {n_param} parameters")


#------Training utilities------

#------EPOCH LOOPS------  

def train_epoch_ae(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
        val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator, intermediate=False, epoch=0):
    
    loss_meter.reset()
    val_loss_meter.reset()
    if performance_meter is not None:
        performance_meter.reset()
        val_performance_meter.reset()

    val_list = []
    i = 0
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
 
        if performance_meter is not None:
            perf = accuracy(X_pred, X)
            performance_meter.update(val=perf, n=X.shape[0])
            performance_meter.add_iter_loss()

        if i % 5000 == 0:
            validate_model(model, validationloader, accelerator, loss_fn, val_loss_meter, val_performance_meter)
            if intermediate:
                with open(log_path+log_file, 'a') as f:
                    if val_performance_meter is not None:
                        f.write(f"\nValidation loss at iteration {i}, tot = {val_loss_meter.sum}, avg = {val_loss_meter.avg}, val perf avg = {val_performance_meter.avg}.")
                    else:
                        f.write(f"\nValidation loss at iteration {i}, tot = {val_loss_meter.sum}, avg = {val_loss_meter.avg}")
            np.savetxt(log_path+"val_loss_iter.csv", val_loss_meter.avg_iter_list)
            np.savetxt(log_path+"train_loss_iter.csv", loss_meter.avg_iter_list)

            if performance_meter is not None:
                np.savetxt(log_path+"train_accuracy_iter.csv", performance_meter.avg_iter_list)
                np.savetxt(log_path+"val_accuracy_iter.csv", val_performance_meter.avg_iter_list)

        #if isinstance(lr_scheduler, (_LRScheduler, OneCycleLR)):
        #    lr_scheduler.step()

        i += 1

    validate_model(model, validationloader, accelerator, loss_fn, val_loss_meter, val_performance_meter)



def train_epoch_cnn(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
            val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator, intermediate=False, epoch=0):

    loss_meter.reset()
    val_loss_meter.reset()
    if performance_meter is not None:
        performance_meter.reset()
        val_performance_meter.reset()

    #lr_list = []
    i = 0
    for X, y in dataloader:
        if accelerator is None:
            X = X.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        y_pred = model(X)
        if performance_meter is None:
            y_pred = y_pred.squeeze()
        loss = loss_fn(y_pred, y)
        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)
        #torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
        loss_meter.update(val=loss.item(), n=X.shape[0])
        loss_meter.add_iter_loss()

        if performance_meter is not None:
            perf = accuracy(y_pred, y)
            performance_meter.update(val=perf, n=X.shape[0])
            performance_meter.add_iter_loss()

        if i % 5000 == 0:
            validate_model(model, validationloader, accelerator, loss_fn, val_loss_meter, val_performance_meter)
            if intermediate:
                with open(log_path+log_file, 'a') as f:
                    if val_performance_meter is not None:
                        f.write(f"\nValidation loss at iteration {i}, tot = {val_loss_meter.sum}, avg = {val_loss_meter.avg}, val perf avg = {val_performance_meter.avg}.")
                    else:
                        f.write(f"\nValidation loss at iteration {i}, tot = {val_loss_meter.sum}, avg = {val_loss_meter.avg}")
            np.savetxt(log_path+"val_loss_iter.csv", val_loss_meter.avg_iter_list)
            np.savetxt(log_path+"train_loss_iter.csv", loss_meter.avg_iter_list)

            if performance_meter is not None:
                np.savetxt(log_path+"train_accuracy_iter.csv", performance_meter.avg_iter_list)
                np.savetxt(log_path+"val_accuracy_iter.csv", val_performance_meter.avg_iter_list)

        #if isinstance(lr_scheduler, (_LRScheduler, OneCycleLR)):
        #    lr_scheduler.step()


        #lr_list.append(lr_scheduler.get_last_lr()[0])
        #lr_state_dict = lr_scheduler.state_dict()
        #if i % 15000 == 0 and i != 0:
        #    lr_state_dict["base_lrs"][0] = lr_state_dict["base_lrs"][0] * 0.1
        #    lr_state_dict["_last_lr"][0] =lr_state_dict["_last_lr"][0] * 0.1
        #    lr_scheduler.load_state_dict(lr_state_dict)

        i += 1

    validate_model(model, validationloader, accelerator, loss_fn, val_loss_meter, val_performance_meter)
    #np.savetxt("/work_dir/220913/mean/lr.csv", lr_list)


def train_epoch_gnn(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
            val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator, intermediate=False, epoch=0):

    loss_meter.reset()
    val_loss_meter.reset()
    if performance_meter is not None:
        performance_meter.reset()
        val_performance_meter.reset()

    model.train()
    i = 0
    for X, data in dataloader:
        device = 'cuda' if accelerator is None else accelerator.device
        optimizer.zero_grad()
        #y_pred, y, weights, _  = model(X, data, device)
        #if weights is not None:
        #    loss = loss_fn(y_pred, y, weights)
        #else:
        y_pred, y, _  = model(X, data, device)
        loss = loss_fn(y_pred, y)
        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
        loss_meter.update(val=loss.item(), n=X.shape[0])    
        loss_meter.add_iter_loss()    

        if performance_meter is not None:
            perf = accuracy(y_pred, y)
            performance_meter.update(val=perf, n=X.shape[0])
            performance_meter.add_iter_loss()

        #if isinstance(lr_scheduler, (_LRScheduler, OneCycleLR)):
        #    lr_scheduler.step()

        if i % 5000 == 0:
            validate_model(model, validationloader, accelerator, loss_fn, val_loss_meter, val_performance_meter)
            if intermediate:
                with open(log_path+log_file, 'a') as f:
                    if val_performance_meter is not None:
                        f.write(f"\nValidation loss at iteration {i}, tot = {val_loss_meter.sum}, avg = {val_loss_meter.avg}, val perf avg = {val_performance_meter.avg}.")
                    else:
                        f.write(f"\nValidation loss at iteration {i}, tot = {val_loss_meter.sum}, avg = {val_loss_meter.avg}")
            #np.savetxt(log_path+"val_loss_iter.csv", val_loss_meter.avg_iter_list)
            #np.savetxt(log_path+"train_loss_iter.csv", loss_meter.avg_iter_list)

            #if performance_meter is not None:
            #    np.savetxt(log_path+"train_accuracy_iter.csv", performance_meter.avg_iter_list)
            #    np.savetxt(log_path+"val_accuracy_iter.csv", val_performance_meter.avg_iter_list)

        
            if accelerator is None or accelerator.is_main_process:
                np.savetxt(log_path+"train_loss.csv", loss_meter.avg_list)
                np.savetxt(log_path+"val_loss.csv", val_loss_meter.avg_list)
                np.savetxt(log_path+"train_loss_iter.csv", loss_meter.avg_iter_list)
                np.savetxt(log_path+"val_loss_iter.csv", val_loss_meter.avg_iter_list)
                if performance_meter is not None:
                    np.savetxt(log_path+"train_perf.csv", performance_meter.avg_list)
                    np.savetxt(log_path+"val_perf.csv", val_performance_meter.avg_list)
                    np.savetxt(log_path+"train_perf_iter.csv", performance_meter.avg_iter_list)
                    np.savetxt(log_path+"val_perf_iter.csv", val_performance_meter.avg_iter_list)
                checkpoint_dict = {
                    "parameters": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    #"loss": loss_meter.avg
                    "epoch": epoch
                    }
                torch.save(checkpoint_dict, log_path+"checkpoint.pth")

        i += 1

    validate_model(model, validationloader, accelerator, loss_fn, val_loss_meter, val_performance_meter)


def train_epoch_gru(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
    val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator):
    
    return train_epoch_cnn(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
        val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator)


#------ TRAIN ------  

def train_model(model, dataloader, loss_fn, optimizer, num_epochs,
        log_path, log_file, train_epoch, accelerator, validate_model, validationloader,
        fine_tuning, lr_scheduler=None, checkpoint_name="checkpoint.pth", loss_name="loss.csv",
        ctd_training=False, checkpoint_ctd="../checkpoint.pth",
        save_interval=1, performance=None, epoch_start=0):
    
    model.train()

    # define average meter objects
    loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    if performance is not None:
        performance_meter = AverageMeter()
        val_performance_meter = AverageMeter()
    else:
        performance_meter = None
        val_performance_meter = None

    # epoch loop
    for epoch in range(epoch_start, epoch_start + num_epochs):
        
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}")
        
        start_time = time.time()
        
        train_epoch(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
            val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator, epoch=epoch)
        
        end_time = time.time()
        
        loss_meter.add_loss()
        val_loss_meter.add_loss()
        if performance is not None:
            performance_meter.add_loss()
            val_performance_meter.add_loss()

        if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.000001:
            lr_scheduler.step()

        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                if performance_meter is None:
                    f.write(f"\nEpoch {epoch+1} completed in {end_time - start_time:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. "+
                            f"Validation loss avg = {val_loss_meter.avg:.4f}")
                else:
                    f.write(f"\nEpoch {epoch+1} completed in {end_time - start_time:.4f} seconds. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}; "+
                            f"performance: {performance_meter.avg:.4f}. Validation loss avg = {val_loss_meter.avg:.4f}; performance: {val_performance_meter.avg:.4f}")

            np.savetxt(log_path+"train_loss.csv", loss_meter.avg_list)
            np.savetxt(log_path+"val_loss.csv", val_loss_meter.avg_list)
            np.savetxt(log_path+"train_loss_iter.csv", loss_meter.avg_iter_list)
            np.savetxt(log_path+"val_loss_iter.csv", val_loss_meter.avg_iter_list)
            if performance is not None:
                np.savetxt(log_path+"train_perf.csv", performance_meter.avg_list)
                np.savetxt(log_path+"val_perf.csv", val_performance_meter.avg_list)
                np.savetxt(log_path+"train_perf_iter.csv", performance_meter.avg_iter_list)
                np.savetxt(log_path+"val_perf_iter.csv", val_performance_meter.avg_iter_list)
            checkpoint_dict = {
                "parameters": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                }
            torch.save(checkpoint_dict, checkpoint_name)

#----- VALIDATION ------

def validate_ae(model, dataloader, accelerator, loss_fn, val_loss_meter, val_performance_meter):

    model.eval()
    with torch.no_grad():
        for X in dataloader:
            if accelerator is None:
                X = X.cuda()
            X_pred = model(X)
            loss = loss_fn(X_pred, X)
            val_loss_meter.update(loss.item(), X.shape[0])
            if val_performance_meter is not None:
                perf = accuracy(X_pred, X)
                val_performance_meter.update(val=perf, n=X.shape[0])
        val_loss_meter.add_iter_loss()
        if val_performance_meter is not None:
            val_performance_meter.add_iter_loss()
    model.train()
    return


def validate_gnn(model, dataloader, accelerator, loss_fn, val_loss_meter, val_performance_meter):

    model.eval()
    with torch.no_grad():
        for X, data in dataloader:
            device = 'cuda' if accelerator is None else accelerator.device
            y_pred, y, _ = model(X, data, device)
            if val_performance_meter is not None:
                perf = accuracy(y_pred, y)
                val_performance_meter.update(val=perf, n=X.shape[0])
            loss = loss_fn(y_pred, y)
            val_loss_meter.update(loss.item(), X.shape[0])
        val_loss_meter.add_iter_loss()
        if val_performance_meter is not None:
            val_performance_meter.add_iter_loss()
    model.train()
    return


#------ TEST ------  

def test_model_ae(model, dataloader, log_path, log_file, accelerator, loss_fn=None, performance=None):
    
    if loss_fn is not None:
        loss_meter = AverageMeter()

    i = 0
    model.eval()
    with torch.no_grad():
        for X in dataloader:
            if accelerator is None:
                X = X.cuda()
            X_pred = model(X)
            loss = loss_fn(X_pred, X) if loss_fn is not None else None
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            if i == 0:
                X_pred = X_pred.detach().cpu().numpy()
                X = X.detach().cpu().numpy()
                with open(log_path+"X_pred.pkl", 'wb') as f:
                    pickle.dump(X_pred, f)
                with open(log_path+"X.pkl", 'wb') as f:
                    pickle.dump(X, f)
            i += 1

    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                    +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}")
    return fin_loss_total, fin_loss_avg

def test_model_cnn(model, dataloader, log_path, log_file, accelerator, loss_fn=None):
    if loss_fn is not None:
        loss_meter = AverageMeter()

    i = 0
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
            if i == 0:
                with open(log_path+log_file, 'a') as f:
                    f.write(f"{list(zip(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()))}")
            i += 1

    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                    +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}")
    return fin_loss_total, fin_loss_avg


def test_model_gnn(model, dataloader, log_path, log_file, accelerator, loss_fn=None, performance=None):
    
    if loss_fn is not None:
        loss_meter = AverageMeter()
    if performance is not None:
        perf_meter = AverageMeter()

    y_pred_list = []
    y_list = []
    model.eval()
    with torch.no_grad():
        for X, data in dataloader:
            device = 'cuda' if accelerator is None else accelerator.device
            y_pred, y, _ = model(X, data, device)
            if loss_fn is not None:
            #    if weights is not None:
            #        loss = loss_fn(y_pred, y, weights)
            #    else:
                loss = loss_fn(y_pred, y)
                loss_meter.update(loss.item(), X.shape[0])
            else:
                loss = None
            if performance is not None:
                perf = accuracy(y_pred, y)
                perf_meter.update(perf, X.shape[0])
                # append results to list
                _ = [y_pred_list.append(yi) for yi in torch.argmax(y_pred, dim=-1).detach().cpu().numpy()]
            else:
                _ = [y_pred_list.append(yi) for yi in y_pred.detach().cpu().numpy()]
            
            _ = [y_list.append(yi) for yi in y.detach().cpu().numpy()]
        y_list = np.array(y_list)
        y_pred_list = np.array(y_pred_list)
        with open(log_path+"y_pred.pkl", 'wb') as f:
            pickle.dump(y_pred_list, f)
        with open(log_path+"y.pkl", 'wb') as f:                  
            pickle.dump(y_list, f)

    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    fin_perf_avg = perf_meter.avg if performance is not None else None

    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                    +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}. Performance = {fin_perf_avg}.")

    return fin_loss_total, fin_loss_avg
   

def test_model_gru(model, dataloader, log_path, log_file, accelerator, loss_fn=None, performance=None):
    
    if loss_fn is not None:
        loss_meter = AverageMeter()
    if performance is not None:
        perf_meter = AverageMeter()

    y_pred_list = []
    y_list = []   
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            if accelerator is None:
                X = X.cuda()
                y = y.cuda()
            y_pred = model(X)
            if performance is None:
                y_pred = y_pred.squeeze()
            # append results to list
            loss = loss_fn(y_pred, y) if loss_fn is not None else None
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            if performance is not None:
                perf = accuracy(y_pred, y)
                perf_meter.update(perf, X.shape[0])
                # append results to list
                _ = [y_pred_list.append(yi) for yi in torch.argmax(y_pred, dim=-1).detach().cpu().numpy()]
            else:
                _ = [y_pred_list.append(yi) for yi in y_pred.detach().cpu().numpy()]

            _ = [y_list.append(yi) for yi in y.detach().cpu().numpy()]
                       
        y_list = np.array(y_list)
        y_pred_list = np.array(y_pred_list)
        with open(log_path+"y_pred.pkl", 'wb') as f:
            pickle.dump(y_pred_list, f)
        with open(log_path+"y.pkl", 'wb') as f:
            pickle.dump(y_list, f)
               
    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    fin_perf_avg = perf_meter.avg if performance is not None else None

    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                    +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}. Performance = {fin_perf_avg}.")
                    
    return fin_loss_total, fin_loss_avg


