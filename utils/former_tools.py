import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime
from distutils.util import strtobool
import pandas as pd

from utils.metrics import metric

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    # if args.decay_fac is None:
    #     args.decay_fac = 0.5
    # if args.lradj == 'type1':
    #     lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    # elif args.lradj == 'type2':
    #     lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    if args.lradj =='type1':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj =='type2':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    elif args.lradj =='type4':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch) // 1))}
    else:
        args.learning_rate = 1e-4
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def vali(model, vali_loader,  criterion, args, device):
    total_loss = []

    model.eval()

    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader.get_iterator()):
            # batch_x = torch.squeeze(batch_x)
            # batch_y = torch.squeeze(batch_y)
            outputs = model(batch_x)
            
            # encoder - decoder
            outputs = outputs[..., 0]
            batch_y = batch_y[..., 0]

            # pred = outputs.detach().cpu()
            # true = batch_y.detach().cpu()
            pred = outputs
            true = batch_y

            # loss = criterion(pred, true)
            loss = masked_mae(pred, true, 0.0)

            total_loss.append(loss)
    # total_loss = np.average(total_loss)
    total_loss = torch.mean(torch.tensor(total_loss))
   

    model.train()

    return total_loss

def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))

def test(model, test_loader, args, device):
    preds = []
    trues = []
    # mases = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader.get_iterator()):
                      
            outputs = model(batch_x)
            
            # encoder - decoder
            outputs = outputs[... , 0]
            batch_y = batch_y[... , 0]

            # pred = outputs.detach().cpu().numpy()
            # true = batch_y.detach().cpu().numpy()
            pred = outputs
            true = batch_y
            
            preds.append(pred)
            trues.append(true)


    # preds = torch.Tensor(preds)
    # trues = torch.Tensor(trues)
    preds = torch.stack(preds[:-1])
    trues = torch.stack(trues[:-1])

    amae = []
    amape = []
    armse = []
    for i in range(args.pred_len):
        pred = preds[..., i]
        real = trues[..., i]
  
        metric = metrics(pred,real)
   
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metric[0], metric[1], metric[2]))
        amae.append(metric[0])
        amape.append(metric[1])
        armse.append(metric[2])


    # return np.mean(amae),np.mean(amape),np.mean(armse)
    return torch.mean(torch.tensor(amae)), torch.mean(torch.tensor(amape)), torch.mean(torch.tensor(armse))



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = (preds-labels)**2
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = torch.abs(preds-labels)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = torch.abs(preds-labels)/labels
    return torch.mean(loss)


def metrics(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse






# # import numpy as np
# def cal_metrics(y_true, y_pred):
#     mse = torch.square(y_pred - y_true)
#     mse = torch.mean(mse)


#     # rmse = torch.square(np.abs(y_pred - y_true))
#     rmse = torch.sqrt(mse)



#     mae = torch.abs(y_pred - y_true)
#     mae = torch.mean(mae)
#     return rmse, 0, mae 



