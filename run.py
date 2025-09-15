from data_provider.data_factory import  data_provider
from utils.former_tools import vali, test, masked_mae, EarlyStopping

from tqdm import tqdm


from models.repst import repst
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings

import argparse
import random
import logging

warnings.filterwarnings('ignore')

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='RePST')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='path_to_data')
parser.add_argument('--data_path', type=str, default='dataset_name')

parser.add_argument('--pred_len', type=int, default=24)
parser.add_argument('--seq_len', type=int, default=24)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.002)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=20)

parser.add_argument('--gpt_layers', type=int, default=9)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.2)

parser.add_argument('--patch_len', type=int, default=6)

parser.add_argument('--stride', type=int, default=7)

parser.add_argument('--tmax', type=int, default=5)


args = parser.parse_args()
device = torch.device(args.device)

logging.basicConfig(filename="./log/{}.log".format(args.data_path), level=logging.INFO)
logging.info(args)

rmses = []
maes = []
mapes = []




train_loader, vali_loader, test_loader = data_provider(args)


time_now = time.time()

model = repst(args, device).to(device)
early_stopping = EarlyStopping(patience=args.patience, verbose=True)
params = model.parameters()
model_optim = torch.optim.Adam(params, lr=args.learning_rate)

# class SMAPE(nn.Module):
#     def __init__(self):
#         super(SMAPE, self).__init__()
#     def forward(self, pred, true):
#         return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
# criterion = SMAPE()
criterion = nn.MSELoss()


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

path = "./checkpoints/{}_{}_{}".format(args.data_path, args.gpt_layers, args.learning_rate)
if not os.path.exists(path):
    os.makedirs(path)
for epoch in range(args.train_epochs):

    iter_count = 0
    train_loss = []
    epoch_time = time.time()
    train_loader.shuffle()
    model_optim.zero_grad()
    for i, (x, y) in enumerate(train_loader.get_iterator()):

        iter_count += 1  
  
        x = x.to(device)
        y = y.to(device)
    
        outputs = model(x)
        outputs = outputs[..., 0]
        y = y[..., 0]
  
        loss = criterion(outputs, y)
 
        train_loss.append(loss.item())

        if i  % 100 == 0:
            print("iters: {},  loss: {}, time_cost: {}".format(i + 1,  np.average(train_loss[-100:]), time.time() - epoch_time))
            logging.info("iters: {},  loss: {}, time_cost: {}".format(i + 1,  np.average(train_loss[-100:]), time.time() - epoch_time))

        loss.backward()

        model_optim.step()
        model_optim.zero_grad()
   
    logging.info("Epoch: {} cost time: {}".format(epoch , time.time() - epoch_time))
    print("Epoch: {} cost time: {}".format(epoch , time.time() - epoch_time))

    train_loss = np.average(train_loss)
    vali_loss = vali(model, vali_loader,  criterion, args, device)
    scheduler.step()
   
    early_stopping(vali_loss, model, path)

    if (epoch + 1) % 1 ==0:

        print("------------------------------------")
        logging.info("------------------------------------")
        mae, mape, rmse = test(model, test_loader, args, device)
        log = 'On average over all horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        logging.info(log.format(mae,mape,rmse))
        print(log.format(mae,mape,rmse))
    
