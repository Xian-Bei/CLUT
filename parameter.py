import argparse
import torch
import numpy as np
import os
import pdb

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_cpu", type=int, default=4, help="for dataloader")
parser.add_argument("--optm", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--lambda_smooth", type=float, default=0.0001, help="smooth regularization strength")
parser.add_argument("--lambda_mn", type=float, default=10.0, help="monotonicity regularization strength")
# train: =1 starts from scratch, >1 load saved checkpoint of epoch-1
# evaluate: evaluate the model of epoch
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--n_epochs", type=int, default=370, help="last epoch of training (include)")
parser.add_argument("--dim", type=int, default=33, help="dimension of 3DLUT")
parser.add_argument("--losses", type=str, default="1*l1 1*cosine", help="one or more loss functions (split by space)")
parser.add_argument("--model", type=str, default="200520", help="model selection")

parser.add_argument("--save_root", type=str, default=".", help="root path to save images/models/logs")
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--data_root", type=str, default="/data", help="root path of data")
parser.add_argument("--dataset", type=str, default="FiveK", help="which dateset to use, dataset class should be implemented first")



cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = "cuda"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
