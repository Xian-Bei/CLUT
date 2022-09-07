import argparse
import numpy as np
import math
import torch
import sys 
import os
import pdb
from os.path import join
import matplotlib.pyplot as plt
# from visualize import *
sys.path.append(".")
# from models import *


def identity3d_tensor(dim): # 3,d,d,d
    step = np.arange(0,dim)/(dim-1) # Double, so need to specify dtype
    rgb = torch.tensor(step, dtype=torch.float32)
    LUT = torch.empty(3,dim,dim,dim)
    LUT[0] = rgb.unsqueeze(0).unsqueeze(0).expand(dim, dim, dim) # r
    LUT[1] = rgb.unsqueeze(-1).unsqueeze(0).expand(dim, dim, dim) # g
    LUT[2] = rgb.unsqueeze(-1).unsqueeze(-1).expand(dim, dim, dim) # b
    return LUT

def identity2d_tensor(dim): # 2,d,d
    # Double, so need to specify dtype
    step = torch.tensor(np.arange(0,dim)/(dim-1), dtype=torch.float32)
    hs = torch.empty(2,dim,dim)
    hs[0] = step.unsqueeze(0).repeat(dim, 1) # r
    hs[1] = step.unsqueeze(1).repeat(1, dim) # g
    return hs
    
def identity1d_tensor(dim): # 1,d
    step = np.arange(0,dim)/(dim-1) # Double, so need to specify dtype
    return torch.tensor(step, dtype=torch.float32).unsqueeze(0)

    

def cube_to_lut(cube): # (n,)3,d,d,d
    if len(cube.shape) == 5:
        to_shape = [
            [0,2,3,1],
            [0,2,1,3],
        ]
    else:
        to_shape = [
            [1,2,0],
            [1,0,2],
        ]
    if isinstance(cube, torch.Tensor):
        lut = torch.empty_like(cube)
        lut[...,0,:,:,:] = cube[...,0,:,:,:].permute(*to_shape[0])
        lut[...,1,:,:,:] = cube[...,1,:,:,:].permute(*to_shape[1])
        lut[...,2,:,:,:] = cube[...,2,:,:,:]
    else:
        lut = np.empty_like(cube)
        lut[...,0,:,:,:] = cube[...,0,:,:,:].transpose(*to_shape[0])
        lut[...,1,:,:,:] = cube[...,1,:,:,:].transpose(*to_shape[1])
        lut[...,2,:,:,:] = cube[...,2,:,:,:]
    return lut

def lut_to_cube(lut): # (n,)3,d,d,d
    if len(lut.shape) == 5:
        to_shape = [
            [0,3,1,2],
            [0,2,1,3],
        ]
    else:
        to_shape = [
            [2,0,1],
            [1,0,2],
        ]

    if isinstance(lut, torch.Tensor):
        cube = torch.empty_like(lut)
        cube[...,0,:,:,:] = lut[...,0,:,:,:].permute(*to_shape[0])
        cube[...,1,:,:,:] = lut[...,1,:,:,:].permute(*to_shape[1])
        cube[...,2,:,:,:] = lut[...,2,:,:,:]
    else:
        cube = np.empty_like(lut)
        cube[...,0,:,:,:] = lut[...,0,:,:,:].transpose(*to_shape[0])
        cube[...,1,:,:,:] = lut[...,1,:,:,:].transpose(*to_shape[1])
        cube[...,2,:,:,:] = lut[...,2,:,:,:]
    return cube
    



def read_3dlut_from_file(file_name, return_type="tensor"):
    file = open(file_name, 'r')
    lines = file.readlines()
    start, end = 0, 0 # 从cube文件读取时
    for i in range(len(lines)):
        if lines[i][0].isdigit() or lines[i].startswith("-"):
            start = i
            break
    for i in range(len(lines)-1,start,-1):
        if lines[i][0].isdigit() or lines[i].startswith("-"):
            end = i
            break
    lines = lines[start: end+1]
    if len(lines) == 262144:
        dim = 64
    elif len(lines) == 35937:
        dim = 33
    else:
        dim = int(np.round(math.pow(len(lines), 1/3)))
    print("dim = ", dim)
    buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)
    # LUT的格式是 cbgr，其中c是 rgb
    # 在lut文件中，一行中依次是rgb
    # r是最先最多变化的，b是变化最少的
    # 往里填的过程中，k是最先最多变化的，它填在最后位置
    for i in range(0,dim):# b
        for j in range(0,dim):# g
            for k in range(0,dim):# r
                n = i * dim*dim + j * dim + k
                x = lines[n].split()
                buffer[0,i,j,k] = float(x[0])# r
                buffer[1,i,j,k] = float(x[1])# g
                buffer[2,i,j,k] = float(x[2])# b

    if return_type in["numpy", "np"]:
        return buffer
    elif return_type in["tensor", "ts"]:
        return torch.from_numpy(buffer)
        # buffer = torch.zeros(3,dim,dim,dim) # 直接用torch太慢了，不如先读入np再直接转torch
    else:
        raise ValueError("return_type should be np or ts")


def from_1d1(v): # n,1,dim  or  n,-1  return n,1,dim,dim,dim
    n = v.shape[0]
    if len(v.shape) == 2: # 需要dim来reshape
        v = v.reshape(n, 1, -1)
    dim = v.shape[2]

    v = v.unsqueeze(-1).unsqueeze(-1) #n,1,d -> n,1,d,1,1
    v = v.expand(n,1,dim,dim,dim)
    return v

def from_3d1(rgb, LUT=None): # n,3,dim  or  n,-1
    n = rgb.shape[0]
    if len(rgb.shape) == 2: # 需要dim来reshape
        rgb = rgb.reshape(n, 3, -1)
    dim = rgb.shape[2]
    if LUT is None:
        LUT = torch.zeros(n, 3, dim, dim, dim).type(rgb.type())
        LUT[:,0] = rgb[:,0].unsqueeze(1).unsqueeze(1).expand(n, dim, dim, dim) # r
        LUT[:,1] = rgb[:,1].unsqueeze(1).unsqueeze(-1).expand(n, dim, dim, dim) # g
        LUT[:,2] = rgb[:,2].unsqueeze(-1).unsqueeze(-1).expand(n, dim, dim, dim) # b
    else:
        LUT[:,0] += rgb[:,0].reshape(n,1,1,dim).expand(n, dim, dim, dim) # r
        LUT[:,1] += rgb[:,1].reshape(n,1,dim,1).expand(n, dim, dim, dim) # g
        LUT[:,2] += rgb[:,2].reshape(n,dim,1,1).expand(n, dim, dim, dim) # b
        
    return LUT

def from_1d2(hs): # n,2,dim,dim  or  n,2,-1  return n,2,dim,dim,dim
    n = hs.shape[0]
    if len(hs.shape) == 2:
        dim = int(np.sqrt(hs.shape[2]))
        hs = hs.reshape(n,2,dim,dim)
    dim = hs.shape[2]

    hs = hs.unsqueeze(2) # n,2,1,dim,dim
    hs = hs.expand(n,2,dim,dim,dim) # 1个 dim*dim 复制为 dim个 dim*dim
    return hs

# hs: n,2,dim,dim  or  n,2,-1
# v: n,1,dim  or  n,-1
def from_1d1_1d2(v, hs): 
    n = v.shape[0]
    if len(v.shape) == 2:
        v = v.reshape(n,1,-1)
        dim = v.shape[2]
    if len(hs.shape) == 2:
        dim = int(np.sqrt(hs.shape[2]))
        hs = hs.reshape(n,2,dim,dim)
    dim = hs.shape[2]

    LUT = torch.empty(n, 3, dim, dim, dim).type(hs.type())
    hs = d2_1(hs)
    LUT[:,:2,...] = hs
    v = d1_1(v) # n,1,d,d,d
    LUT[:,2,...] = v[:,0]
    return LUT


# def gamma3d_tensor(dim, gamma=1): # 3,d,d,d
#     step = np.arange(0,dim)/(dim-1) # Double, so need to specify dtype
#     step = np.power(step, 1/gamma)
#     rgb = torch.tensor(step, dtype=torch.float32)
#     LUT = torch.empty(3,dim,dim,dim)
#     LUT[0] = rgb.unsqueeze(0).unsqueeze(0).expand(dim, dim, dim) # r
#     LUT[1] = rgb.unsqueeze(-1).unsqueeze(0).expand(dim, dim, dim) # g
#     LUT[2] = rgb.unsqueeze(-1).unsqueeze(-1).expand(dim, dim, dim) # b

# def wb3d_tensor(dim, wb=[0.82943738, 1.02267336, 1.2246885]): # 3,d,d,d
#     step = np.arange(0,dim)/(dim-1) # Double, so need to specify dtype
#     rgb = torch.tensor(step, dtype=torch.float32)
#     LUT = torch.empty(3,dim,dim,dim)
#     LUT[0] = rgb*wb[0].unsqueeze(0).unsqueeze(0).expand(dim, dim, dim) # r
#     LUT[1] = rgb*wb[1].unsqueeze(-1).unsqueeze(0).expand(dim, dim, dim) # g
#     LUT[2] = rgb*wb[2].unsqueeze(-1).unsqueeze(-1).expand(dim, dim, dim) # b
#     return LUT