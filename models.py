import os
import pdb
import math
import trilinear
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from utils.LUT import *
import time

class CLUTNet(nn.Module): 
    def __init__(self, nsw, dim, *args, **kwargs):
        super(CLUTNet, self).__init__()
        self.pre = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        last_channel = 128
        self.backbone = BackBone(last_channel)

        self.classifier = nn.Sequential(
                nn.Conv2d(last_channel, 128,1,1),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Conv2d(128, int(nsw[:2]),1,1),
            )
        
        num = int(nsw[:2])
        if len(nsw)==2:
            s = -1
            w = -1
        else:
            s = int(nsw[2:4])
            w = int(nsw[4:])
        self.CLUTs = CLUT(int(nsw[:2]),dim,s,w)
        self.TrilinearInterpolation = TrilinearInterpolation()


    def forward(self, img, img_org, TVMN=None, *args, **kwargs):
        pre = time.time()
        feature = self.backbone(self.pre(img))
        stage1 = (time.time() - pre)*1000

        pre = time.time()
        weight = self.classifier(feature)[:,:,0,0] # n, num
        LUT, tvmn = self.CLUTs(weight, TVMN)
        stage2 = (time.time() - pre)*1000

        pre = time.time()
        # pdb.set_trace()
        img_res = self.TrilinearInterpolation(LUT, img_org)
        stage3 = (time.time() - pre)*1000

        return img_res + img_org , {
            "LUT": LUT,
            "tvmn": tvmn,
            "time_cost": [stage1,stage2,stage3]
        }


class CLUT(nn.Module):
    def __init__(self, num, dim=33, s="-1", w="-1", *args, **kwargs):
        super(CLUT, self).__init__()
        self.num = num
        self.dim = dim
        self.s,self.w = s,w = eval(str(s)), eval(str(w))
        # +: compressed;  -: uncompressed
        if s == -1 and w == -1: # standard 3DLUT
            self.mode = '--'
            self.LUTs = nn.Parameter(torch.zeros(num,3,dim,dim,dim))
        elif s != -1 and w == -1:  
            self.mode = '+-'
            self.s_Layers = nn.Parameter(torch.rand(dim, s)/5-0.1)
            self.LUTs = nn.Parameter(torch.zeros(s, num*3*dim*dim))
        elif s == -1 and w != -1: 
            self.mode = '-+'
            self.w_Layers = nn.Parameter(torch.rand(w, dim*dim)/5-0.1)
            self.LUTs = nn.Parameter(torch.zeros(num*3*dim, w))

        else: # full-version CLUT
            self.mode = '++'
            self.s_Layers = nn.Parameter(torch.rand(dim, s)/5-0.1)
            self.w_Layers = nn.Parameter(torch.rand(w, dim*dim)/5-0.1)
            self.LUTs = nn.Parameter(torch.zeros(s*num*3,w))
        print("n=%d s=%d w=%d"%(num, s, w), self.mode)

    def reconstruct_luts(self):
        dim = self.dim
        num = self.num
        if self.mode == "--":
            D3LUTs = self.LUTs
        else:
            if self.mode == "+-":
                # d,s  x  s,num*3dd  -> d,num*3dd -> d,num*3,dd -> num,3,d,dd -> num,-1
                CUBEs = self.s_Layers.mm(self.LUTs).reshape(dim,num*3,dim*dim).permute(1,0,2).reshape(num,3,self.dim,self.dim,self.dim)
            if self.mode == "-+":
                # num*3d,w x w,dd -> num*3d,dd -> num,3ddd
                CUBEs = self.LUTs.mm(self.w_Layers).reshape(num,3,self.dim,self.dim,self.dim)
            if self.mode == "++":
                # s*num*3, w  x   w, dd -> s*num*3,dd -> s,num*3*dd -> d,num*3*dd -> num,-1
                CUBEs = self.s_Layers.mm(self.LUTs.mm(self.w_Layers).reshape(-1,num*3*dim*dim)).reshape(dim,num*3,dim**2).permute(1,0,2).reshape(num,3,self.dim,self.dim,self.dim)
            D3LUTs = cube_to_lut(CUBEs)

        return D3LUTs

    def combine(self, weight, TVMN): # n,num
        dim = self.dim
        num = self.num

        # if TVMN is None and self.mode == "++":
        # if False:
        #     tvmn = 0 

        #     clut = (weight.reshape(1,-1,1)*self.LUTs.reshape(self.s,self.num,3*self.w)).sum(1).reshape(self.s*3,self.w) # s,3*w
        #     cube = self.s_Layers.mm(clut.mm(self.w_Layers).reshape(-1,3*dim*dim)).reshape(dim,3,dim**2).permute(1,0,2).reshape(1,3,self.dim,self.dim,self.dim)
        #     d3lut = cube
        #     # d3lut = cube_to_lut(cube)
        # else:
        D3LUTs = self.reconstruct_luts()
        if TVMN is None:
            tvmn = 0
        else:
            tvmn = TVMN(D3LUTs)
        d3lut = weight.mm(D3LUTs.reshape(num,-1)).reshape(-1,3,dim,dim,dim)
        return d3lut, tvmn

    def forward(self, weight, TVMN=None):
        lut, tvmn = self.combine(weight, TVMN)
        return lut, tvmn

class BackBone(nn.Module): 
    def __init__(self, last_channel=128, ): # org both
        super(BackBone, self).__init__()
        ls = [
            *discriminator_block(3, 16, normalization=True), # 128**16
            *discriminator_block(16, 32, normalization=True), # 64**32
            *discriminator_block(32, 64, normalization=True), # 32**64
            *discriminator_block(64, 128, normalization=True), # 16**128
            *discriminator_block(128, last_channel, normalization=False), # 8**128
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(1),
        ]
        self.model = nn.Sequential(*ls)
        
    def forward(self, x):
        return self.model(x)


class TVMN(nn.Module): # (n,)3,d,d,d     (n,)3,d
    def __init__(self, dim=33, mode='new'):
        super(TVMN,self).__init__()
        self.dim = dim
        self.relu = torch.nn.ReLU()
       
        weight_r = torch.ones(1, 1, dim, dim, dim - 1, dtype=torch.float)
        weight_r[..., (0, dim - 2)] *= 2.0
        weight_g = torch.ones(1, 1, dim, dim - 1, dim, dtype=torch.float)
        weight_g[..., (0, dim - 2), :] *= 2.0
        weight_b = torch.ones(1, 1, dim - 1, dim, dim, dtype=torch.float)
        weight_b[..., (0, dim - 2), :, :] *= 2.0        
        self.register_buffer('weight_r', weight_r, persistent=False)
        self.register_buffer('weight_g', weight_g, persistent=False)
        self.register_buffer('weight_b', weight_b, persistent=False)

        self.register_buffer('tvmn_shape', torch.empty(3), persistent=False)
        self.mode = mode
        if mode =="old":
            print("old tvmn")
        elif mode == "new":
            print("new tvmn")


    def forward(self, LUT): 
        dim = self.dim
        tvmn = 0 + self.tvmn_shape
        if len(LUT.shape) > 3: # n,3,d,d,d  or  3,d,d,d
            if self.mode == 'old':
                '''old wrong
                # 以r为例，任何两个格子，当只有r坐标不同，其他（b和g）坐标都相同时，这两个格子的rgb三个值都要光滑和非减
                '''
                dif_r = LUT[...,:-1] - LUT[...,1:]
                dif_g = LUT[...,:-1,:] - LUT[...,1:,:]
                dif_b = LUT[...,:-1,:,:] - LUT[...,1:,:,:]
                tvmn[0] =   torch.mean(dif_r**2 * self.weight_r[:,0]) + \
                            torch.mean(dif_g**2 * self.weight_g[:,0]) + \
                            torch.mean(dif_b**2 * self.weight_b[:,0])
                tvmn[1] =   torch.mean(self.relu(dif_r * self.weight_r[:,0])**2) + \
                            torch.mean(self.relu(dif_g * self.weight_g[:,0])**2) + \
                            torch.mean(self.relu(dif_b * self.weight_b[:,0])**2)
                tvmn[2] = 0
            elif self.mode == 'new':
                # 以r为例，任何两个格子，当只有r坐标不同，其他（b和g）坐标都相同时，这两个格子的r值都要光滑和单调, g和b不一定要单调，只要光滑就行
                # 例如 (2,x,x) -> (4,x,x); (3,x,x) -> (3,x,x)   not ok
                # (2,x,x) -> (4,x,x); (3,x,x) -> (5,x-1,x-1)  ok
                dif_r = LUT[...,0,:,:,:-1] - LUT[...,0,:,:,1:] # (n,)d,d ,d-1
                dif_g = LUT[...,1,:,:-1,:] - LUT[...,1,:,1:,:]
                dif_b = LUT[...,2,:-1,:,:] - LUT[...,2,1:,:,:]
                tvmn[0] =   torch.mean(dif_r**2 * self.weight_r[:,0]) + \
                            torch.mean(dif_g**2 * self.weight_g[:,0]) + \
                            torch.mean(dif_b**2 * self.weight_b[:,0])
                tvmn[1] =   torch.mean(self.relu(dif_r * self.weight_r[:,0])**2) + \
                            torch.mean(self.relu(dif_g * self.weight_g[:,0])**2) + \
                            torch.mean(self.relu(dif_b * self.weight_b[:,0])**2)

                dif_r = LUT[...,(1,2),:,:,:-1] - LUT[...,(1,2),:,:,1:] # (n,)d,d,d-1
                dif_g = LUT[...,(0,2),:,:-1,:] - LUT[...,(0,2),:,1:,:]
                dif_b = LUT[...,(0,1),:-1,:,:] - LUT[...,(0,1),1:,:,:]
                tvmn[2] =   torch.mean(self.relu(dif_r * self.weight_r)**2) + \
                            torch.mean(self.relu(dif_g * self.weight_g)**2) + \
                            torch.mean(self.relu(dif_b * self.weight_b)**2)
            else:
                raise
        else: # n,3,d  or  3,d
            dif = LUT[...,:-1] - LUT[...,1:]
            tvmn[1] = torch.mean(self.relu(dif))
            dif = dif**2
            dif[...,(0,dim-2)] *= 2.0
            tvmn[0] = torch.mean(dif)
            tvmn[2] = 0
            # tv2, mn2 = self.forward(from_3d1(LUT))
            # pdb.set_trace()
        return tvmn


def discriminator_block(in_filters, out_filters, kernel_size=3, sp="2_1", normalization=False):
    stride = int(sp.split("_")[0])
    padding = int(sp.split("_")[1])

    layers = [
        nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2),
    ]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))

    return layers


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        if batch == 1:
            assert 1 == trilinear.forward(lut,
                                          x,
                                          output,
                                          dim,
                                          shift,
                                          binsize,
                                          W,
                                          H,
                                          batch)
        elif batch > 1:
            output = output.permute(1, 0, 2, 3).contiguous()
            assert 1 == trilinear.forward(lut,
                                          x.permute(1,0,2,3).contiguous(),
                                          output,
                                          dim,
                                          shift,
                                          binsize,
                                          W,
                                          H,
                                          batch)
            output = output.permute(1, 0, 2, 3).contiguous()

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])

        if batch == 1:
            assert 1 == trilinear.backward(x,
                                           x_grad,
                                           lut_grad,
                                           dim,
                                           shift,
                                           binsize,
                                           W,
                                           H,
                                           batch)
        elif batch > 1:
            assert 1 == trilinear.backward(x.permute(1,0,2,3).contiguous(),
                                           x_grad.permute(1,0,2,3).contiguous(),
                                           lut_grad,
                                           dim,
                                           shift,
                                           binsize,
                                           W,
                                           H,
                                           batch)
        return lut_grad, x_grad

# trilinear_need: imgs=nchw, lut=3ddd or 13ddd
class TrilinearInterpolation(torch.nn.Module):
    def __init__(self, mo=False, clip=False):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        
        if lut.shape[0] > 1:
            if lut.shape[0] == x.shape[0]: # n,c,H,W
                res = torch.empty_like(x)
                for i in range(lut.shape[0]):
                    res[i:i+1] = TrilinearInterpolationFunction.apply(lut[i:i+1], x[i:i+1])[1]
            else:
                n,c,h,w = x.shape
                # pdb.set_trace()
                res = torch.empty(n, lut.shape[0], c, h, w).cuda()
                for i in range(lut.shape[0]):
                    res[:,i] = TrilinearInterpolationFunction.apply(lut[i:i+1], x)[1]
        else: # n,c,H,W
            res = TrilinearInterpolationFunction.apply(lut, x)[1]
        return res
        # return torch.clip(TrilinearInterpolationFunction.apply(lut, x)[1],0,1)
