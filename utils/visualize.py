
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
import math

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

channel_ls = ['r', 'g', 'b']


def draw_matrix_weak(matrix_weak, title=None, point_size=50, save_dir=None): # matrix_weak： (n,)1,d,d  or  (n,)3,d,d
    fullscreen()
    dim = matrix_weak.shape[-1]
    matrix_weak *= dim
    x, y = np.arange(0,dim), np.arange(0,dim)
    x, y = np.meshgrid(x, y)
    if len(matrix_weak.shape) == 3:
        matrix_weak = np.expand_dims(matrix_weak, 0)
    n = matrix_weak.shape[0]
    n = 10 # how many lut to view
    col = 5 # how many columns to view
    row = int(n/col)
    plt.subplots_adjust(wspace=0.,hspace=0)
    for i in range(n):
        ax = plt.subplot(row,col,i+1, projection='3d')
        ax.set_title(title)
        ax.set_zlim(0,dim) 
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_zticks([]) 
        # ax.set_xlabel('R')
        # ax.set_ylabel('G')
        # ax.set_zlabel("B")
        for c in range(matrix_weak.shape[1]):
            ax.scatter(x,y,matrix_weak[i,c],c=matrix_weak[i,c],s=point_size)
    
    if save_dir is not None:
        plt.pause(1)
        plt.savefig(os.path.join(save_dir, "matrix_weak.pdf", format="pdf"))
        plt.clf()
    else:
        plt.show()


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

def draw_strong(luts, title=None, save_dir=None, time=1): # luts: (n,3,d,d,d)

    #n,1,dim
    def draw_curve(curves, ax, color='b'):
        dim = curves.shape[1]
        x = np.arange(dim)
        for j in range(0, curves.shape[0], 100):
            ax.plot(x, curves[j], c=color, rasterized=False)


    if not isinstance(luts, list) and len(luts.shape) < 5:
        luts = [luts]
    if isinstance(luts, list):
        n = len(luts)
    else:
        n = luts.shape[0]
    step = 15
    dim = luts[0].shape[2]

    n = min(n, 5) # how many to visualize 
    fig, axes = plt.subplots(3, n ,sharex="col", sharey="row")
    plt.subplots_adjust(hspace=0.3)
    fullscreen()
    for lut_idx in range(n):
        lut = luts[lut_idx]
        lut *= dim
        cube = lut_to_cube(lut) # 3,ddd
        cube = cube.permute(2,3,0,1).reshape(-1,3,dim).detach().cpu().numpy()
        for c in range(3):
            ax = axes[c, lut_idx]
            title = "$\phi_{%d}^{%c}$"%(lut_idx+1, ['r','g','b'][c])
            ax.set_title(title, y=1, fontsize="xx-large")
            ax.set_box_aspect(1)
            ax.set_xticks(range(0,dim,10))
            if lut_idx == int(n/2) and c == 2:
                ax.set_xlabel("$c_{in}$", fontsize="xx-large", labelpad=8)
            if lut_idx == 0 and c == 1:
                ax.set_ylabel("$c_{out}$", fontsize="xx-large", labelpad=11)
            draw_curve(cube[::1,c,], ax=ax, color=channel_ls[c])

    if save_dir is not None:
        plt.pause(time)
        plt.savefig(os.path.join(save_dir, "S.pdf"), format="pdf")
    else:
        plt.show()


def draw_weak(luts, title=None, time=2, point_size=40, save_dir=None): # luts: (n,3,d,d,d)

    import matplotlib.patches as mpatches
    import matplotlib.transforms as mtransforms
    import copy
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    def add_top_cax(ax_ls, pad, width):
        cax_ls = []
        for ax in ax_ls:
            axpos = ax.get_position()
            caxpos = mtransforms.Bbox.from_extents(
                axpos.x0,
                axpos.y0 + pad,
                axpos.x1,
                axpos.y1 + pad + width
            )
            cax = ax.figure.add_axes(caxpos)
            cax.set_visible(False)
            cax_ls.append(cax)

        return cax_ls

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
    else:
        col = n
    step = 15
    dim = luts[0].shape[2]

    x, y = np.arange(0,dim), np.arange(0,dim)
    x, y = np.meshgrid(x, y)
    min_, max_ = 100, -100
    for lut_idx in range(n):
        
        lut = luts[lut_idx]
        lut *= dim
        cube = lut_to_cube(lut)
        
        for dim_idx in range(0,dim,step):
            for channel_idx in range(len(channel_ls)):
                ax = plt.subplot(3,n,(channel_idx)*col+lut_idx+1,projection='3d')
                title = "$\phi_{%d}^{%c}$"%(lut_idx+1, channel_ls[channel_idx])
                ax.set_title(title, y=0.99, fontsize="xx-large")
                # ax.set_zlim(0,dim)
                if lut_idx == n-1:
                    ax.set_zticks(range(0,dim,step))
                    if channel_idx == 1:
                        ax.set_zlabel("$c_{out}$",labelpad=6,fontsize="xx-large")
                else:
                    ax.set_zticks([])
                if channel_idx == 2:
                    ax.set_xticks([0,10,20,30])
                    ax.set_yticks([0,10,20,30])
                    ax.set_xlabel("$x_{in}$",labelpad=6,fontsize="xx-large")
                    ax.set_ylabel("$y_{in}$",labelpad=6,fontsize="xx-large")
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                c = cube[channel_idx,dim_idx,::2,::2].flatten() - dim_idx
                if c.min()<min_:
                    min_ = c.min()
                if c.max()>max_:
                    max_ = c.max()
                ax.scatter(x[::2,::2], y[::2,::2], cube[channel_idx,dim_idx,::2,::2], c=c, s=point_size, rasterized=False)  # 绘制数据点
    
    ls = add_top_cax([plt.subplot(3,n,i+1) for i in range(0,n)], pad=0.085, width=0.02)
    cmap1 = copy.copy(cm.viridis)
    norm1 = mcolors.Normalize(vmin=min_, vmax=max_)
    im1 = cm.ScalarMappable(norm=norm1, cmap=cmap1)
    bar = plt.colorbar(im1, ax=ls, location='top')

    if save_dir is not None:
        plt.pause(time)
        plt.savefig(os.path.join(save_dir, "W (%d).pdf"%point_size), format="pdf")
        plt.clf()
    else:
        plt.show()  

# Please see show3D.ipynb for faster 3D visualization
def draw_3D(lut, title=None, point_size=30, time=2): # lut: (1,d,d,d)  OR  (3,d,d,d)
    fullscreen()
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
    ax.set_zlabel("R")
    ax.scatter(x, y, z, c=lut, s=point_size)
    plt.show()


if __name__ == "__main__":

    root = "/mnt/tvmn1/input"
    name = "Test+20-1-1_models/"
    epoch = 181
    model = torch.load(os.path.join(root, name, "model{:0>4}.pth".format(epoch)),map_location=torch.device('cpu'))
    luts = cube_to_lut(model['CLUTs.LUTs'].reshape(-1,3,33,33,33))
    draw_strong(luts+identity3d_tensor(33).unsqueeze(0), save_dir=None) 
    draw_weak(luts+identity3d_tensor(33).unsqueeze(0), save_dir=None)
    # draw_3D(luts[0]+identity3d_tensor(33))


    root = "/mnt/tvmn1/input/"
    name = "Test+20-120_models"
    epoch = 354
    model = torch.load(os.path.join(root, name, "model{:0>4}.pth".format(epoch)),map_location=torch.device('cpu'))
    matrix_weak = model['CLUTs.w_Layers'].numpy().reshape(-1,1,33,33)
    draw_matrix_weak(matrix_weak)