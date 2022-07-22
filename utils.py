import torch
import os

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
        self.avg_list.append(self.avg)

def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def train_epoch_ae(model, dataloader, loss_fn, optimizer, loss_meter,
                   performance_meter, performance, device, lr_scheduler): # note: I've added a generic performance to replace accuracy
    for X, _ in dataloader:
        X = X.to(device)
        optimizer.zero_grad() 
        target = model(X)
        loss = loss_fn(X, target)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        #acc = performance(X, target)
        loss_meter.update(val=loss.item(), n=X.shape[0])

def train_model_ae(model, dataloader, loss_fn, optimizer, num_epochs, log_path, checkpoint_loc=None,
                checkpoint_name="checkpoint.pt", performance=None, lr_scheduler=None,
                device=None, lr_scheduler_step_on_epoch=False):

    # create the folder for the checkpoints (if it's not None)
    if checkpoint_loc is not None:
        os.makedirs(checkpoint_loc, exist_ok=True)

    if device is None:
        device = use_gpu_if_possible()
    
    model = model.to(device)
    model.train()

    # epoch loop
    for epoch in range(num_epochs):

        loss_meter = AverageMeter()
        performance_meter = AverageMeter()

        with open(log_path+'log_input.txt', 'a') as f:
            f.write(f"Epoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.5f}")
        
        lr_scheduler_batch = lr_scheduler if not lr_scheduler_step_on_epoch else None

        train_epoch_ae(model, dataloader, loss_fn, optimizer, loss_meter, performance_meter, performance,
                    device, lr_scheduler_batch)

        with open(log_path+'log_ae.txt', 'a') as f:
            f.write(f"Epoch {epoch+1} completed. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.4f}.")
        
        # produce checkpoint dictionary -- but only if the name and folder of the checkpoint are not None
        if checkpoint_name is not None and checkpoint_loc is not None:
            checkpoint_dict = {
                "parameters": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_loc, checkpoint_name))
        
        if lr_scheduler is not None and lr_scheduler_step_on_epoch:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(loss_meter.avg)
            else:
                lr_scheduler.step()

    return loss_meter.sum, loss_meter.avg_list #, performance_meter.avg

def test_model(model, dataloader, performance=None, loss_fn=None, device=None):
    # create an AverageMeter for the loss if passed
    if loss_fn is not None:
        loss_meter = AverageMeter()
    
    if device is None:
        device = use_gpu_if_possible()

    model = model.to(device)

    #performance_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            target = model(X)
            loss = loss_fn(X, target) if loss_fn is not None else None
            #acc = performance(y_hat, y)
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            #performance_meter.update(acc, X.shape[0])
    # get final performances
    fin_loss = loss_meter.sum if loss_fn is not None else None
    #fin_perf = performance_meter.avg
    print(f"TESTING - loss {fin_loss if fin_loss is not None else '--'}") #" - performance {fin_perf:.4f}")
    return fin_loss #fin_perf