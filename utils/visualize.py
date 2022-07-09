
import math
import torch
import pdb
import os
from os.path import join
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append("./utils")
from LUT import *
from scipy.linalg import orth
from numpy.linalg import matrix_rank

channels = ['r', 'g', 'b']
xlabels = ['$g_{in}$', '$b_{in}$', '$g_{in}$']
ylabels = ['$n_{in}$', '$r_{in}$', '$r_{in}$']

def fullscreen():
    plt.get_current_fig_manager().full_screen_toggle()
    # plt.get_current_fig_manager().window.showMaximized()
    # plt.get_current_fig_manager().frame.Maximize(True)

    
    # Option 1
    # QT backend
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()

    # Option 2
    # TkAgg backend
    # manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())

    # Option 3
    # WX backend
    # manager = plt.get_current_fig_manager()
    # manager.frame.Maximize(True)



# (n,)c,d
def draw_curve(rgb, ax=None, title=None, save_dir=None, time=0.1, c=None):
    print(rgb.shape)
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    # rgb = np.squeeze(rgb) 
    if ax is None:
        ax = plt
        show = save_dir is None
    else:
        show = False
    if title:
        ax.title(title)
    # ax.ylim((-0.1,0.1))
    if len(rgb.shape) == 2:# c,dim
        rgb = np.expand_dims(rgb, 0)
    dim = rgb.shape[2]
    # idt = identity1d_tensor(dim).squeeze().numpy()
    x = np.arange(dim)
    # width = [5,3,1]
    width = [1,1,1]
    channel_ls = ['r','g','b']
    # pdb.set_trace()
    # for i in range(rgb.shape[0]):
    if rgb.shape[1] == 1:
        if c is None:
            c = 'b'
        # pdb.set_trace()
        idt = np.array(range(0,33)).reshape(1,1,33)
        idx = np.abs(rgb-idt).max(2).flatten()
        idx = np.argsort(idx)
        # for j in range(rgb.shape[0]):
        for j in range(10):
            j = idx[-j-1]
            ax.plot(x, rgb[j,0], c=c, rasterized=False)
        ax.plot(x, rgb[idx[0],0], c=c, rasterized=False)
    else:
        for j in range(rgb.shape[0]):
            for c in range(rgb.shape[1]):
                
                # ax.plot(x, rgb[i,c], c=channel_ls[c], label=channel_ls[c], linewidth=width[c])
                ax.plot(x, rgb[j,c], c=channel_ls[c], linewidth=width[c], rasterized=False)
                # ax.plot(x, rgb[j,c], linewidth=width[c], rasterized=False)

    # plt.legend()
    if save_dir is not None:
        ax.savefig(save_dir)
        ax.close()
    elif show:
        ax.pause(time)
        # plt.pause(0.00001)
        ax.clf()

import math
#  20,(n),c,d
def draw1D(luts, title=None, save_dir=None, time=0.1):
    if not isinstance(luts, list) and len(luts.shape) < 5:
        luts = [luts]
    if isinstance(luts, list):
        n = len(luts)
    else:
        n = luts.shape[0]
    step = 15
    dim = luts[0].shape[2]
    idt = identity3d_tensor(dim).numpy()

    n = 5
    fig, axes = plt.subplots(3, n ,sharex="col", sharey="row")  # 创建一个三维的绘图工程
    plt.subplots_adjust(hspace=0.3)
    fullscreen()
    for lut_idx in range(n):
        lut = luts[lut_idx]
        lut += idt
        lut *= dim
        cube = lut_to_cube(lut) # 3,ddd
        cube = cube.permute(2,3,0,1).reshape(-1,3,dim)
        # ax = axes[int(lut_idx/row), lut_idx%row]
        for c in range(3):
            ax = axes[c, lut_idx]
            title = "$\phi_{%d}^{%c}$"%(lut_idx+1, ['r','g','b'][c])
            ax.set_title(title, y=1, fontsize="xx-large")
            ax.set_box_aspect(1)
            ax.set_xticks(range(0,dim,10))
            if lut_idx == 2 and c == 2:
                ax.set_xlabel("$c_{in}$", fontsize="xx-large", labelpad=8)
            if lut_idx == 0 and c == 1:
                ax.set_ylabel("$c_{out}$", fontsize="xx-large", labelpad=11)
            draw_curve(cube[::1,c:c+1,], ax=ax, time=1, c=channels[c])
    
    plt.pause(time)
    plt.savefig(os.path.join(save_dir, "V-in.pdf"), format="pdf")
    # plt.show()

def restore_in(LUTs, in_Layers):
    num = 20
    dim = 33
    p = 20
    q = 20
    # num,p,q  -> num,q,p -> num*q, p  x  p,3*d  -> num*q,3*d -> num,q,3*d -> num,3*d,q 
    return LUTs.permute(0,2,1).reshape(-1,p).mm(in_Layers).reshape(num, q, 3*dim).permute(0,2,1)
def restore_inter(LUTs, inter_Layers):
    dim = 33
    num = 20
    p = 20
    q = inter_Layers.shape[0]
    # num,3*d,q  -> num*3*d,q x  q,d*d  -> num*3*d,d*d -> num,3d,dd
    return LUTs.reshape(-1,q).mm(inter_Layers)
def main_draw_curve():
    idt = identity3d_tensor(33).unsqueeze(0)
    # print(idt.shape)

    # name = "Test+20_fiveK_models"
    # epoch = 383
    # luts = torch.load(name + "_model{:0>4}_LUTs0.pth".format(epoch)).reshape(20,3,33,33,33) + idt
    # cubes = lut_to_cube(luts)
    # curves = cubes.permute(0,3,4,1,2).reshape(luts.shape[0],33*33,3,33)

    name = "/mnt/notvmn/input/Test+202020_models/model"
    epoch = 307
    m = torch.load(name + "{:0>4}.pth".format(epoch),map_location=torch.device('cpu'))
    res_in = restore_in(m["LUT_model.LUTs"], m["LUT_model.P_Layers"]) # 20,3*33,q
    # curves = res_in.permute(0,2,1).reshape(20,-1,3,33)
    # res_inter = restore_inter(m["LUT_model.LUTs"], m["LUT_model.Q_Layers"])
    cubes = restore_inter(res_in, m["LUT_model.Q_Layers"]).reshape(20,3,33,33,33) 
    cubes += lut_to_cube(idt)
    print(matrix_rank(cubes.reshape(-1,33*33)))
    curves = cubes.permute(0,3,4,1,2).reshape(cubes.shape[0],33*33,3,33)
    orth_data = orth(curves.reshape(-1,3*33))
    pdb.set_trace()
    draw_curve(curves,time=10)
    # for i in range(cubes.shape[0]):
    #     draw_curve(cubes[i])
    # draw_curve(cubes,time=1,save_dir="../results/V-in")
    # bases = np.load("bases of p 0.9.npy")
    # draw_curve(bases.reshape(-1,3,33), time=3, save_dir="../results/in-pca") 
    # bases = np.load("new_data p 0.95.npy")
    # draw_curve(bases.reshape(bases.shape[0],1,-1)[:30], time=3, save_dir="../results/p-sparsity") 


# def drawwhat(luts, title=None, time=10, point_size=30):
#     if not isinstance(luts, list):
#         luts = [luts]
#     rows = len(luts)*2
#     dim = luts[0].shape[2]
#     idt = identity3d_tensor(dim).numpy()*dim

#     for lut_idx, lut in enumerate(luts):
#         # pdb.set_trace()
#         if len(lut.shape) == 5: # 1,3,d,d,d
#             lut = lut.squeeze()
#         lut *= dim
#         luts[lut_idx] = lut_to_cube(lut).transpose(2,3,0,1).reshape(-1,3,dim)
#         # from_3d1(torch.from_numpy(rgb).unsqueeze(0)).squeeze().numpy()
#         # luts[lut_idx] *= 400

#     # 固定 x_in, x_out 随另外连个维度输入的变化而变化
#     x, y = np.arange(0,dim), np.arange(0,dim)
#     x, y = np.meshgrid(x, y)
#     for lut_idx, lut in enumerate(luts):
#         # pdb.set_trace()
#         draw_curve(lut, title=title,time=time)
#             # for j in range(0,dim):
#     plt.clf()


# def draw2D(luts, title=None, time=1, point_size=30, save_dir=None):
#     fullscreen()
#     if not isinstance(luts, list) and len(luts.shape) < 5:
#         luts = [luts]
#     if isinstance(luts, list):
#         n = len(luts)
#     else:
#         n = luts.shape[0]
#     col = 5
#     if n > col:
#         n = col
#     step = 15
#     dim = luts[0].shape[2]
#     idt = identity3d_tensor(dim).numpy()

#     # for lut_idx in range(n):
#         # ax = plt.subplot(rows,3,3*lut_idx+4)
#         # rgb = np.zeros((3,dim))
#         # rgb[0] = lut[0].mean((0,1))
#         # rgb[1] = lut[1].mean((0,2))
#         # rgb[2] = lut[2].mean((1,2))
#         # draw_curve(rgb, ax)

#         # luts[lut_idx] = lut_to_cube(lut)
#         # from_3d1(torch.from_numpy(rgb).unsqueeze(0)).squeeze().numpy()
#         # luts[lut_idx] *= 400

#     x, y = np.arange(0,dim), np.arange(0,dim)
#     x, y = np.meshgrid(x, y)
#     for lut_idx in range(n):
        
#         lut = luts[lut_idx]
#         lut += idt
#         # lut[lut<0]=0
#         # lut[lut>1]=1
#         lut *= dim
#         cube = lut_to_cube(lut)
        
#         # plt.suptitle(title)
#         # fig, axes = plt.subplots(n, 3, sharex="col", sharey="row", projection='3d')  # 创建一个三维的绘图工程
#         for dim_idx in range(0,dim,step):
#             for channel_idx in range(len(channels)):
#                 ax = plt.subplot(n,3,lut_idx*3+channel_idx+1, projection='3d')  # 创建一个三维的绘图工程
#                 title = "$\phi_{%d}^{%c}$"%(lut_idx+1, channels[channel_idx])
#                 ax.set_title(title, y=0.99, fontsize="xx-large")
#                 # ax.set_zlim(0,dim)
#                 if channel_idx == 2:
#                     ax.set_zticks(range(0,dim,step))
#                     ax.set_zlabel("$c_{out}$",labelpad=6,fontsize="xx-large")  # 坐标轴
#                 else:
#                     ax.set_zticks([])
#                     # ax.set_zlabel("$%s_{out}$"%channels[channel_idx],labelpad=-8)  # 坐标轴
#                 if lut_idx == n-1:
#                     ax.set_xticks([0,10,20,30])
#                     ax.set_yticks([0,10,20,30])
#                     ax.set_xlabel("$x_{in}$",labelpad=6,fontsize="xx-large")
#                     ax.set_ylabel("$y_{in}$",labelpad=6,fontsize="xx-large")
#                     # ax.set_xlabel(xlabels[channel_idx],labelpad=6,fontsize="xx-large")
#                     # ax.set_ylabel(ylabels[channel_idx],labelpad=6,fontsize="xx-large")
#                 else:
#                     ax.set_xticks([])
#                     ax.set_yticks([])
#                     # ax.set_xlabel(xlabels[channel_idx],labelpad=-8)
#                     # ax.set_ylabel(ylabels[channel_idx],labelpad=-8)
#                 c = cube[channel_idx,dim_idx].flatten()/dim
#                 # if channel_idx == 0:
#                 #     c = [[1,0,0,x] for x in c]
#                 # elif channel_idx == 1:
#                 #     c = [(0,1,0,x) for x in c]
#                 # elif channel_idx == 2:
#                 #     c = [(0,0,1,x) for x in c]
#                 ax.scatter(x, y, cube[channel_idx,dim_idx], c=c, s=point_size)  # 绘制数据点

#     plt.show()

#     # plt.pause(time)
#     # plt.savefig(os.path.join(save_dir, "V-inter.pdf"), format="pdf")
#     # # plt.clf()

def draw2D(luts, title=None, time=1, point_size=20, save_dir=None):
    fullscreen()
    if not isinstance(luts, list) and len(luts.shape) < 5:
        luts = [luts]
    if isinstance(luts, list):
        n = len(luts)
    else:
        n = luts.shape[0]
    col = 5
    if n > col:
        n = col
    step = 15
    dim = luts[0].shape[2]
    idt = identity3d_tensor(dim).numpy()

    # for lut_idx in range(n):
        # ax = plt.subplot(rows,3,3*lut_idx+4)
        # rgb = np.zeros((3,dim))
        # rgb[0] = lut[0].mean((0,1))
        # rgb[1] = lut[1].mean((0,2))
        # rgb[2] = lut[2].mean((1,2))
        # draw_curve(rgb, ax)

        # luts[lut_idx] = lut_to_cube(lut)
        # from_3d1(torch.from_numpy(rgb).unsqueeze(0)).squeeze().numpy()
        # luts[lut_idx] *= 400

    x, y = np.arange(0,dim), np.arange(0,dim)
    x, y = np.meshgrid(x, y)
    for lut_idx in range(n):
        
        lut = luts[lut_idx]
        lut += idt
        # lut[lut<0]=0
        # lut[lut>1]=1
        lut *= dim
        cube = lut_to_cube(lut)
        
        # plt.suptitle(title)
        # fig, axes = plt.subplots(n, 3, sharex="col", sharey="row", projection='3d')  # 创建一个三维的绘图工程
        for dim_idx in range(0,dim,step):
            for channel_idx in range(len(channels)):
                ax = plt.subplot(3,n,(channel_idx)*col+lut_idx+1,projection='3d')  # 创建一个三维的绘图工程
                title = "$\phi_{%d}^{%c}$"%(lut_idx+1, channels[channel_idx])
                ax.set_title(title, y=0.99, fontsize="xx-large")
                # ax.set_zlim(0,dim)
                if lut_idx == n-1:
                    ax.set_zticks(range(0,dim,step))
                    ax.set_zlabel("$c_{out}$",labelpad=6,fontsize="xx-large")  # 坐标轴
                else:
                    ax.set_zticks([])
                    # ax.set_zlabel("$%s_{out}$"%channels[channel_idx],labelpad=-8)  # 坐标轴
                if channel_idx == 2:
                    ax.set_xticks([0,10,20,30])
                    ax.set_yticks([0,10,20,30])
                    ax.set_xlabel("$x_{in}$",labelpad=6,fontsize="xx-large")
                    ax.set_ylabel("$y_{in}$",labelpad=6,fontsize="xx-large")
                    # ax.set_xlabel(xlabels[channel_idx],labelpad=6,fontsize="xx-large")
                    # ax.set_ylabel(ylabels[channel_idx],labelpad=6,fontsize="xx-large")
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # ax.set_xlabel(xlabels[channel_idx],labelpad=-8)
                    # ax.set_ylabel(ylabels[channel_idx],labelpad=-8)
                c = cube[channel_idx,dim_idx,].flatten()/dim
                ax.scatter(x, y, cube[channel_idx,dim_idx,], c=c, s=point_size, rasterized=True)  # 绘制数据点
                # c = cube[channel_idx,dim_idx,::2,::2].flatten()/dim
                # ax.scatter(x[::2,::2], y[::2,::2], cube[channel_idx,dim_idx,::2,::2], c=c, s=point_size, rasterized=True)  # 绘制数据点

    # plt.show()

    plt.pause(0.1)
    plt.savefig(os.path.join(save_dir, "V-inter(%d).pdf"%point_size), format="pdf")
    plt.clf()

def draw2D2(luts, title=None, time=1, point_size=30):
    if not isinstance(luts, list):
        luts = [luts]
    step = 1
    dim = luts[0].shape[2]
    if dim >40:
        step = 2
    for lut_idx, lut in enumerate(luts):
        if len(lut.shape) == 5: # 1,3,d,d,d
            lut = lut.squeeze()
        lut *= dim
        rgb = np.zeros((3,dim))
        rgb[0] = lut[0].mean((0,1))
        rgb[1] = lut[1].mean((0,2))
        rgb[2] = lut[2].mean((1,2))
        luts[lut_idx] = lut - from_3d1(torch.from_numpy(rgb).unsqueeze(0)).squeeze().numpy()
        # luts[lut_idx] *= 100
    # 固定 x_in, x_out 随另外连个维度输入的变化而变化
    x, y = np.arange(0,dim), np.arange(0,dim)
    x, y = np.meshgrid(x, y)
    rows = len(luts)
    for dim_idx in range(0,dim,step):
        plt.suptitle(title)
        for lut_idx, lut in enumerate(luts):
            if len(lut.shape) == 5:
                lut = lut.squeeze()
            for channel_idx in range(len(channels)):
                ax = plt.subplot(rows,3,channel_idx+1+3*lut_idx, projection='3d')  # 创建一个三维的绘图工程
                ax.set_zlim(0,dim) 
                ax.set_zlabel(channels[channel_idx] + ' output value')  # 坐标轴
                # if channel_idx == 2:
                #     ax.set_title("{} value when R = {}".format(channels[channel_idx], dim_idx))
                #     ax.set_xlabel('G')
                #     ax.set_ylabel('B')
                #     ax.scatter(x, y, lut[channel_idx,:,:,dim_idx]*dim, c=lut[channel_idx][:,:,dim_idx], s=point_size)  # 绘制数据点
                # else:
                if True:
                    ax.set_title("{} value when B = {}".format(channels[channel_idx], dim_idx))
                    ax.set_xlabel('R')
                    ax.set_ylabel('G')
                    ax.scatter(x, y, lut[channel_idx,dim_idx], c=lut[channel_idx][dim_idx], s=point_size)  # 绘制数据点
        
        plt.pause(time)
        plt.clf()


def draw_layers(layers, title=None, time=10, point_size=50): # (n,)1,d,d  or  (n,)3,d,d
    fullscreen()
    dim = layers.shape[-1]
    layers *= dim
    x, y = np.arange(0,dim), np.arange(0,dim)
    x, y = np.meshgrid(x, y)
    if len(layers.shape) == 3:
        layers = np.expand_dims(layers, 0)
    n = layers.shape[0]
    n = 10
    col = 5
    row = int(n/col)
    # pdb.set_trace()
    plt.subplots_adjust(wspace=0.,hspace=0)
    for i in range(n):
        ax = plt.subplot(row,col,i+1, projection='3d')  # 创建一个三维的绘图工程
        ax.set_title(title)
        ax.set_zlim(0,dim) 
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_zticks([]) 
        # ax.set_xlabel('R')
        # ax.set_ylabel('G')
        # ax.set_zlabel("B")  # 坐标轴
        for c in range(layers.shape[1]):
            ax.scatter(x,y,layers[i,c],c=layers[i,c],s=point_size)
    # plt.subplots_adjust(left=0.1,
    #                     bottom=0.1,
    #                     right=0.2,
    #                     top=0.2,
    #                     wspace=0.2,
    #                     hspace=0.2)
    # plt.pause(1)
    # plt.savefig(os.path.join("../results", "M-inter.pdf"), format="pdf")
    # plt.savefig(os.path.join("../results", "pca-inter.pdf"), format="pdf")

    plt.show()



def draw3D(lut, title=None, point_size=30, time=0.2): # 1,ddd  OR  3,ddd
    # lut = 1-lut
    # print(lut.shape)
    if len(lut.shape) == 5:
        lut = lut.squeeze()
    if isinstance(lut, torch.Tensor):
        lut = lut.numpy()
    dim = lut.shape[1]

    x, y, z = np.arange(0,dim), np.arange(0,dim), np.arange(0,dim)
    x, y, z = np.meshgrid(x, y, z)
    lut = np.clip(lut,0,1).transpose(1,2,3,0)
    if lut.shape[3] == 1:
        lut = lut.repeat(3,3)
    lut = lut.reshape(-1,lut.shape[3])
    ax = plt.subplot(111, projection='3d')
    ax.set_title(title)

    ax.set_xlabel('G')
    ax.set_ylabel('B')
    ax.set_zlabel("R")  # 坐标轴
    ax.scatter(x, y, z, c=lut, s=point_size)  # 绘制数据点
    plt.pause(time)


def draw_inverse(lut, title="", time=1, point_size=1): # 3,d,d,d
    if len(lut.shape) == 5:
        lut = lut.squeeze()
    if isinstance(lut, torch.Tensor):
        lut = lut.numpy()
    dim = lut.shape[1]
    color = np.clip(lut,0,1).transpose(1,2,3,0)
    color = color.reshape(-1,color.shape[3])
    lut.reshape(3,-1)
    ax = plt.subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('G')
    ax.set_ylabel('B')
    ax.set_zlabel("R")  # 坐标轴
    ax.scatter(lut[0], lut[1], lut[2], c=color, s=point_size)  # 绘制数据点
    plt.pause(time)
    plt.clf()

# def mian_draw_weak():

def main_draw_layers():
    dim = 33

    # root = "/mnt/tvmn1/input/"
    # name = "Test+20-120_models"
    # epoch = 354
    # model = torch.load(os.path.join(root, name, "model{:0>4}.pth".format(epoch)),map_location=torch.device('cpu'))
    # layers = model['LUT_model.inter_Layers'].numpy()
    # layers = layers.reshape(-1,1,dim,dim)
    # ###########################################
    layers = np.load("20_bases/bases of q 20.npy")
    # layers = np.load("10_bases/bases of q 30.npy")
    layers = layers.reshape(-1,1,dim,dim)
    
    
    # for i in range(layers.shape[0]):
    draw_layers(layers)

def main_draw_strong_weak():
    
    #################################################
    
    root = "/mnt/tvmn1/input"
    name = "Test+10-1-1_models/"
    epoch = 335

    # root = "/mnt/tvmn1/input"
    # name = "Test+200515_models/"
    # epoch = 315

    # root = "/mnt/tvmn2/input"
    # name = "Test+10-1-1+33+tvmn2_models/"
    # epoch = 200

    # root = "/mnt/notvmn/input"
    # name = "Test+20-1-1_models"
    # epoch = 280
    

    model = torch.load(os.path.join(root, name, "model{:0>4}.pth".format(epoch)),map_location=torch.device('cpu'))
    luts = cube_to_lut(model['LUT_model.LUTs'].reshape(-1,3,33,33,33))
    ######################################################

    # name = "model0293 0293_0.0000--25.4170_0.7721--0.8843_best"
    # key = 'LUT_model.LUTs'
    # model = torch.load(name + ".pth", map_location=torch.device('cpu')) #没法用GPU加速
    # luts = model[key].reshape(-1,3,33,33,33)

    ######################################################
    # name = "Test+20_fiveK_models"
    # epoch = 383
    # luts = torch.load(name + "_model{:0>4}_LUTs0.pth".format(epoch))
    ######################################################


    # draw2D(luts, time=0.01, save_dir="../results")
    draw1D(luts, save_dir="../results") 
    # time=0.01, save_dir="../results")
    
    # for i in range(layers.shape[0]):
    #     draw2D2(layers[i])
    #     drawwhat([lut],ls[i]+"--%d"%i)
    #     pdb.set_trace()
    #     for j in range(32):
    #         draw_layers(lut[0,:,j],ls[i]+"--%d"%i)




if __name__ == "__main__":
    main_draw_strong_weak()
    # main_draw_layers()

    # main_draw_curve()

    
    # idt = identity3d_tensor(32).numpy()
    # id1 = identity1d_tensor(8).expand(64,8).reshape(8,8,8).permute(2,0,1)
    ######################################################### 看cubes
    # cubes = np.load("3D-dct%d-dim32.npy"%(6**3))
    # cube_num = cubes.shape[0]
    # for i in range(150,cube_num):
    #     print(i)
    #     # pdb.set_trace()
    #     draw3D(-cubes[i:i+1],title=str(i))

    ######################################################## 看学的layers
    # mat = (torch.rand(32**2, 32, 1, 1)/5-0.1).numpy()
    # root = "abl_models"
    # name = "Fenkai+layer_5++L36D17+meanvar"
    # epoch = 16
    # model = torch.load(os.path.join(root, name, "ckp{:0>4}.pth".format(epoch)))['model']
    # mat = model['layers'].cpu().numpy() # d*d,l,1,1
    # mat = np.load("sb.npy")

    # if len(mat.shape)>2:
    #     layer_num, dim = mat.shape[1], int(np.sqrt(mat.shape[0]))
    #     print(layer_num)
    #     mat = mat.reshape(dim,dim,layer_num)
    #     for i in range(layer_num):
    #         draw_layers(mat[:,:,i], "%d"%(i))
    # else:
    #     layer_num, dim = mat.shape[0], int(np.sqrt(mat.shape[1]))
    #     print(layer_num)
    #     mat = mat.reshape(layer_num,dim,dim)
    #     for i in range(layer_num):
    #         draw_layers(mat[i], "%d"%(i))
    ######################################################### 看curves
    # root = "ABL33_models"
    # name = "ABL+3*3D+32"
    # epoch = 30
    # dim = 17
    # model = torch.load(os.path.join(root, name, "model{:0>4}.pth".format(epoch)))
    # curves = model['rgb_model.1.weight'].reshape(3,dim,-1).cpu().numpy()
    # curves_bias = model['rgb_model.1.bias'].reshape(3,dim).cpu().numpy()
    # mean = curves.mean(2)
    # draw_curve(mean)
    # var = 0
    # for i in range(curves.shape[2]):
    #     var += np.power(curves[:,:,i]-mean,2).mean()
    #     # draw_curve(curves[:,:,i], title=str(i))
    # pdb.set_trace()
    




    # mat = np.load("dct64-dim32.npy")
    # draw2D1(id1.unsqueeze(0).expand(3,8,8,8).contiguous())
    # lut = torch.empty(3, 8,8,8)
    # lut[0] = id1.permute(1,2,0)
    # lut[1] = id1.permute(1,0,2)
    # lut[2] = id1
    # draw2D1(lut)
    
    '''1'''

    # weight = np.load("weight1l*64_best.npy") # lut_num, 3, dim, layer_num, 1,1
    # mat = np.load("layer1l*64_best.npy") # 1,1,layer_num,d,d
    # # weight = np.load("tp/1_l=d_15195_27.601/weight1-l=d.npy") # lut_num, 3, dim, layer_num, 1,1
    # # mat = np.load("tp/1_l=d_15195_27.601/mat1-l=d.npy") # 1,1,layer_num,d,d
    # lut_num, layer_num, dim = weight.shape[0], weight.shape[3], weight.shape[2]
    # for i in range(63,0,-1):
    #     draw_layers(mat[:,:,i], i)
    # for i in range(11, lut_num):
    #     lut = (weight[i] * mat).sum(2) + idt
    #     # draw_inverse(lut, "%d"%i)
    #     draw_inverse(lut, "%d"%i)
    #     # for j in range(3):
    #     #     for k in range(0,dim,10):
    #     #         layer = (weight[i,j,k] * mat[0,0]).sum(0)
    #     #         # pdb.set_trace()
    #     #         # draw_curve(weight[i,j,k,:,0,0])
    #     #         i = 10
    #     #         j = 2
    #     #         draw_layer(layer, "%d_%d_%d"%(i,j,k))

    '''2'''
    # weight = np.load("tp/2_l=d_8192_27.216/weight2-l=d.npy") # lut_num, dim, layer_num, 1,1,1
    # mat = np.load("tp/2_l=d_8192_27.216/mat2-l=d.npy") # 1,layer_num,3,d,d
    # lut_num, layer_num, dim = weight.shape[0], weight.shape[2], weight.shape[1]
    # # for i in range(layer_num):
    # #     draw_layers(mat[0,i], i)
    # print(weight.shape)
    # print(mat.shape)
    # for i in range(11, lut_num):
    #     lut = (weight[i] * mat).sum(1)
    #     # .transpose(0,1) 
    #     # draw2D3(lut, "%d"%i)
    #     for j in range(dim):
    #         layers = lut[j]
    #         draw_layers(layers)
        #     for k in range(0,dim,10):
        #         # pdb.set_trace()
        #         # draw_curve(weight[i,j,k,:,0,0])
        #         i = 10
        #         j = 2
        #         draw_layer(layer, "%d_%d_%d"%(i,j,k))

    '''3'''
    # weight = np.load("tp/weight3-l=3.npy") # lut_num, 3, layer_num, 1,1,1
    # mat = np.load("tp/mat3-l=3.npy") # 1,layer_num,d,d,d
    # lut_num, layer_num, dim = weight.shape[0], weight.shape[2], mat.shape[2]
    # for i in range(layer_num):
    #     draw3D(mat[:,i])


# 展示当 X channel 上输入值固定，输出值随另外两个channel输入值改变的关系
# 输入为3DLUT列表，每行展示一个
# （可选）同时展示 X channel 上输出值随输入值的大致变化曲线（将另外两个轴上取平均）
# def draw2D(luts, title=None, time=1, point_size=30):
#     if not isinstance(luts, list):
#         luts = [luts]
#     rows = len(luts)*2
#     step = 5
#     dim = luts[0].shape[2]
#     if dim > 40:
#         step = 5
#     idt = identity3d_tensor(dim).numpy()*dim

#     for lut_idx, lut in enumerate(luts):
#         if len(lut.shape) == 5: # 1,3,d,d,d
#             lut = lut.squeeze()
#         lut *= dim
#         # ax = plt.subplot(rows,3,3*lut_idx+4)
#         # rgb = np.zeros((3,dim))
#         # rgb[0] = lut[0].mean((0,1))
#         # rgb[1] = lut[1].mean((0,2))
#         # rgb[2] = lut[2].mean((1,2))
#         # draw_curve(rgb, ax)
#         luts[lut_idx] = lut
#         #  - idt
#         # from_3d1(torch.from_numpy(rgb).unsqueeze(0)).squeeze().numpy()
#         # luts[lut_idx] *= 400

#     # 固定 x_in, x_out 随另外连个维度输入的变化而变化
#     x, y = np.arange(0,dim), np.arange(0,dim)
#     x, y = np.meshgrid(x, y)
#     for dim_idx in range(0,dim,step):
#         plt.suptitle(title)
#         for lut_idx, lut in enumerate(luts):
#             for channel_idx in range(len(channels)):
#                 ax = plt.subplot(rows,3,3*lut_idx+channel_idx+1, projection='3d')  # 创建一个三维的绘图工程
#                 ax.set_title("{} = ".format(channels[channel_idx]) + ','.join([str(i) for i in range(0,dim,step)]))
#                 ax.set_zlim(0,dim)
#                 if channel_idx == 0:# R
#                     ax.set_xlabel('G')
#                     ax.set_ylabel('B')
#                     ax.set_zlabel(channels[channel_idx] + ' output value')  # 坐标轴
#                     ax.scatter(x, y, lut[0,:,:,dim_idx], c=lut[channel_idx][:,:,dim_idx], s=point_size)  # 绘制数据点
#                 elif channel_idx == 1:# G
#                     ax.set_xlabel('B')
#                     ax.set_ylabel('R')
#                     ax.set_zlabel(channels[channel_idx] + ' output value')  # 坐标轴
#                     ax.scatter(x, y, lut[1,:,dim_idx,:], c=lut[channel_idx][:,dim_idx,:], s=point_size)  # 绘制数据点
#                 else:# B
#                     ax.set_xlabel('G')
#                     ax.set_ylabel('R')
#                     ax.set_zlabel(channels[channel_idx] + ' output value')  # 坐标轴
#                     ax.scatter(x, y, lut[2,dim_idx], c=lut[channel_idx][dim_idx], s=point_size)  # 绘制数据点

#         plt.pause(time)
#     plt.clf()
    