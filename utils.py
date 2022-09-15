import numpy as np
import time
import sys
import pickle

import torch
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


def weighted_mse_loss(input_batch, target_batch, device='cuda'):
    weights = torch.tensor([1, 2, 5, 10, 20, 50]).to(device)
    weight = torch.ones((target_batch.shape), device=device)
    for i, target in enumerate(target_batch):
        weight[i] = weights[0] if target <= 2 else weights[1] if target <= 5 else weights[2] if target <= 10 else weights[3] if target <= 20 else weights[5]
        #weight[i] = weights[0] if target <= 0.01 else weights[1] if target <= 0.1 else weights[2] if target <= 0.5 else weights[3] if target <= 1 else weights[4] if target <= 5 else weights[5]
    return (weight * (input_batch - target_batch) ** 2).sum() / weight.sum()

def mse_loss_mod(input_batch, target_batch, alpha=0.25,  device='cuda'):
    return ((input_batch - target_batch) ** 2).sum() / input_batch.shape[0] + alpha * ((torch.log(input_batch+10e-9) - torch.log(target_batch+10e-9)) ** 2).sum() / input_batch.shape[0]

def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def load_encoder_checkpoint(model, checkpoint, log_path, log_file, accelerator, fine_tuning=True, net_name='encoder'):
    if accelerator is None or accelerator.is_main_process:
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

def train_epoch_ae(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
        val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator, intermediate=False):
    
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

        #if isinstance(lr_scheduler, (_LRScheduler, OneCycleLR)):
        #    lr_scheduler.step()

        i += 1

    validate_model(model, validationloader, accelerator, loss_fn, val_loss_meter, val_performance_meter)



def train_epoch_cnn(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
            val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator, intermediate=False):

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
            val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator, intermediate=False):

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
        y_pred, y = model(X, data, device)
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
            np.savetxt(log_path+"val_loss_iter.csv", val_loss_meter.avg_iter_list)
            np.savetxt(log_path+"train_loss_iter.csv", loss_meter.avg_iter_list)

            if performance_meter is not None:
                np.savetxt(log_path+"train_accuracy_iter.csv", performance_meter.avg_iter_list)

        i += 1

    validate_model(model, validationloader, accelerator, loss_fn, val_loss_meter, val_performance_meter)


def train_epoch_gru(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
    val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator):
    
    return train_epoch_cnn(model, dataloader, loss_fn, optimizer, lr_scheduler, loss_meter, performance_meter, val_loss_meter,
        val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator)


#------ TRAIN ------  

def train_model(model, dataloader, loss_fn, optimizer, num_epochs,
        log_path, log_file, train_epoch, accelerator, validate_model, validationloader,
        lr_scheduler=None, checkpoint_name="checkpoint.pth", loss_name="loss.csv",
        ctd_training=False, checkpoint_ctd="../checkpoint.pth",
        save_interval=1, performance=None):
    
    epoch_start = 0

    if ctd_training:
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write("\nLoading the checkpoint to continue the training.")
        checkpoint = torch.load(checkpoint_ctd)
        model.load_state_dict(checkpoint["parameters"])
        #optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_start = checkpoint["epoch"] + 1                       #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #loss = checkpoint["loss"]
    
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
            val_performance_meter, log_path, log_file, validationloader, validate_model, accelerator)
        
        end_time = time.time()
        
        loss_meter.add_loss()
        val_loss_meter.add_loss()
        if performance is not None:
            performance_meter.add_loss()
            val_performance_meter.add_loss()

        if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.000001:
            #if not isinstance(lr_scheduler, (_LRScheduler, OneCycleLR)):
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
                #"loss": loss_meter.avg
                }
            torch.save(checkpoint_dict, checkpoint_name)

    if accelerator is None or accelerator.is_main_process:
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
            "loss": loss_meter.avg             
            }
        torch.save(checkpoint_dict, checkpoint_name)
    
        return loss_meter.sum, loss_meter.avg_list


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

def validate_gru(model, dataloader, accelerator, loss_fn, val_loss_meter, val_performance_meter):

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            if accelerator is None:    
                X = X.cuda()
                y = y.cuda()
            y_pred = model(X).squeeze()
            if val_performance_meter is not None:
                # for cross entropy loss                                            <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                #y_pred = torch.where(y_pred > 0.5, 1, 0)
                #y_pred = y_pred.to(torch.int64)
                #y = y.to(torch.int64)
                perf = accuracy(y_pred, y)
                val_performance_meter.update(val=perf, n=X.shape[0])
            loss = loss_fn(y_pred, y)
            val_loss_meter.update(loss.item(), X.shape[0])
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
            y_pred, y = model(X, data, device)
            if val_performance_meter is not None:
                # for cross entropy loss                                                <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                #y_pred = torch.where(y_pred > 0.5, 1, 0)
                #y_pred = y_pred.to(torch.int64)
                #y = y.to(torch.int64)
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
                sys.exit()
                with open(log_path+log_file, 'a') as f:
                    f.write(f"{list(zip(X, X_pred))}")
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

    i = 0
    model.eval()
    with torch.no_grad():
        for X, data in dataloader:
            device = 'cuda' if accelerator is None else accelerator.device
            y_pred, y = model(X, data, device)
            loss = loss_fn(y_pred, y) if loss_fn is not None else None
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            if performance is not None:
                perf = accuracy(y_pred, y)
                perf_meter.update(perf, X.shape[0])
                # for cross entropy loss                        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                y_pred = torch.where(y_pred > 0.5, 1, 0)
                y_pred = y_pred.to(torch.int64)
                y = y.to(torch.int64)
            if i == 0:
                y_pred = y_pred.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                if performance is not None:
                    y_pred = np.where(y_pred > 0.5, 1, 0)
                    y_pred, y = y_pred.astype(int), y.astype(int)
                    equal = np.where(y_pred == y)
                    different = np.where(y_pred != y)
                with open(log_path+log_file, 'a') as f:
                    f.write(f"\n\n\n{list(zip(y[equal], y_pred[equal]))}")
                    f.write(f"\n\n\n{list(zip(y[different], y_pred[different]))}")
            i += 1

    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            if performance is not None:
                f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                        +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}. Accuracy {perf_meter.avg}.")
            else:
                f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                        +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}") 
    return fin_loss_total, fin_loss_avg 

def test_model_gru(model, dataloader, log_path, log_file, accelerator, loss_fn=None):
    if loss_fn is not None:
        loss_meter = AverageMeter()
    
    #y_pred_list = []
    #y_list = []
    model.eval()
    with torch.no_grad():
        i = 0
        for X, y in dataloader:
            #print(y)
            #y_list = [y_list.append(yi) for yi in y.numpy()]
            if accelerator is None:
                X = X.cuda()
                y = y.cuda()
            y_pred = model(X).squeeze()
            #print(y_pred)
            #y_pred_list = [y_pred_list.append(yi) for yi in y_pred.cpu().numpy()]
            loss = loss_fn(y_pred, y) if loss_fn is not None else None
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            if i == 0:
                with open(log_path+log_file, 'a') as f: 
                    f.write(f"{list(zip(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()))}")
            i += 1
    
    #print(y_list)
    #y_list = torch.tensor(y_list)
    #y_pred_list = torch.tensor(y_pred_list)
    #R2 = r2_score(y_pred_list, y_list)
    fin_loss_total = loss_meter.sum if loss_fn is not None else None
    fin_loss_avg = loss_meter.avg if loss_fn is not None else None
    if accelerator is None or accelerator.is_main_process:
        with open(log_path+log_file, 'a') as f:
            f.write(f"\nTESTING - loss total = {fin_loss_total if fin_loss_total is not None else '--'},"
                    +f"loss avg = {fin_loss_avg if fin_loss_avg is not None else '--'}")
                    #+f" R2 = {R2}")
    return fin_loss_total, fin_loss_avg

    #return test_model_cnn(model, dataloader, log_path, log_file, accelerator, loss_fn)
