import argparse
import torch
import numpy as np
import os
from ipdb import set_trace as S
from time import time


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=4, help="for dataloader")
parser.add_argument("--optm", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--tvmn", default=False, action="store_true", help="whether no use tv and mn constrain")

# --epoch for train:  =1 starts from scratch, >1 load saved checkpoint of <epoch-1>
# --epoch for eval:   load the model of <epoch> and evaluate
parser.add_argument("--epoch", type=int, default=1)

parser.add_argument("--num_epochs", type=int, default=400, help="last epoch of training (include)")
parser.add_argument("--losses", type=str, nargs="+", default=["l1", "cos"], help="one or more loss functions")

parser.add_argument("--model", type=str, nargs="+", default=["CLUTNet", "20+05+20", 33], help="model configuration, [n+s+w, dim]")
# parser.add_argument("--model", type=str, nargs="+", default=['HashLUT', '7+13'], help="model configuration, [nt, res, mlp, backbone]") 

parser.add_argument("--name", type=str, help="name for this training (if None, use <model> instead)")

parser.add_argument("--save_root", type=str, default=".", help="root path to save images/models/logs")
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--data_root", type=str, default="/data", help="root path of data")

parser.add_argument("--dataset", type=str, default="FiveK", help="which dateset class to use (should be implemented first)")



np.set_printoptions(suppress=True)
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = "cuda" if cuda else 'cpu'
