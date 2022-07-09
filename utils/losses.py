
import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb
import math
import numpy as np
from typing import Tuple

def l1(fake, expert, weight=1):
    return (fake - expert).abs().mean()*weight

def psnr(fake, expert):
    # pdb.set_trace()
    mse = (fake - expert).pow(2).mean()
    if mse.pow(2) == 0:
        mse += 1e-6
    if torch.max(expert) > 2:
        max_ = 255.
    else:
        max_ = 1.
    return 10 * torch.log10(max_**2 / (mse)) 

def cosine(fake, expert, weight=1): # 余弦距离 -1~0~1，对应角度180~90~0，我们希望余弦距离越小越好，so 1-
    return (1 - torch.nn.functional.cosine_similarity(fake, expert, 1)).mean()*weight

