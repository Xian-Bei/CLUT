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

# epoch for train:  =1 starts from scratch, >1 load saved checkpoint of <epoch-1>
# epoch for eval:   load the model of <epoch> and evaluate
parser.add_argument("--epoch", type=int, default=1)

parser.add_argument("--n_epochs", type=int, default=370, help="last epoch of training (include)")
parser.add_argument("--dim", type=int, default=33, help="dimension of 3DLUT")
parser.add_argument("--losses", type=str, default="1*l1 1*cosine", help="one or more loss functions (splited by space)")
parser.add_argument("--model", type=str, default="20+05+20", help="model configuration, n+s+w")

parser.add_argument("--save_root", type=str, default="../", help="root path to save images/models/logs")
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--data_root", type=str, default="/data", help="root path of data")

# Dataset Class should be implemented first for different dataset format")
parser.add_argument("--dataset", type=str, default="FiveK", help="which dateset to use")



cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = "cuda"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
