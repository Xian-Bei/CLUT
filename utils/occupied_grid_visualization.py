
import torch
from os.path import join
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
import cv2
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import ipdb
channel_ls = ['r', 'g', 'b']

def fullscreen():
    plt.get_current_fig_manager().full_screen_toggle()
    
def voxelize(ax, dim, inp):
    fc = [1,1,1,0.8]
    lc = [75/256,102/256,130/256]
    lw = 2
    
    voxelarray = np.ones((dim-1,dim-1,dim-1))
    
    inp_floor = np.floor(inp).astype(np.int8)
    # voxelarray[inp_floor[0],inp_floor[1],inp_floor[2]] = 0
    fc = np.zeros((dim-1,dim-1,dim-1,4)) + np.array(fc).reshape(1,1,1,4)
    fc[inp_floor[0],inp_floor[1],inp_floor[2]] = [0,0,0,0.4]
    sha = ax.voxels(voxelarray, facecolors=fc, edgecolor=lc, shade=False, linewidth=lw)
    ax.set_box_aspect([1,1,1])
    
def count_one_image(path, dim=33):
    img = cv2.imread(path, -1).reshape(-1,3)
    img = img/256*dim if img.max() < 256 else img/65536*dim
    flor = np.floor(img)
    ceil = np.ceil(img)
    neibors = [0,1,2,3,4,5,6,7]
    neibors[0] = n000 = np.concatenate((flor[:,0:1],flor[:,1:2],flor[:,2:3]), 1)
    neibors[1] = n001 = np.concatenate((flor[:,0:1],flor[:,1:2],ceil[:,2:3]), 1)
    neibors[2] = n010 = np.concatenate((flor[:,0:1],ceil[:,1:2],flor[:,2:3]), 1)
    neibors[3] = n011 = np.concatenate((flor[:,0:1],ceil[:,1:2],ceil[:,2:3]), 1)
    neibors[4] = n100 = np.concatenate((ceil[:,0:1],flor[:,1:2],flor[:,2:3]), 1)
    neibors[5] = n101 = np.concatenate((ceil[:,0:1],flor[:,1:2],ceil[:,2:3]), 1)
    neibors[6] = n110 = np.concatenate((ceil[:,0:1],ceil[:,1:2],flor[:,2:3]), 1)
    neibors[7] = n111 = np.concatenate((ceil[:,0:1],ceil[:,1:2],ceil[:,2:3]), 1)
    all_neibors = np.concatenate(neibors, 0)
    unique_neibors = np.unique(all_neibors,axis=0)
    proportion = unique_neibors.shape[0]/(dim**3)
    # ax = plt.subplot(1,1,1,projection='3d')
    # ax.scatter(unique_neibors[:,0],unique_neibors[:,1],unique_neibors[:,2])
    # plt.show()  
    return unique_neibors, proportion

    
    
def draw3DLUT(ax, lut, title=None, point_size=1200): # 1,ddd  OR  3,ddd
    
    if len(lut.shape) == 5:
        lut = lut.squeeze()
    if isinstance(lut, torch.Tensor):
        lut = lut.numpy()
    lut = np.clip(lut,0,1).transpose(1,2,3,0)
    if lut.shape[3] == 1:
        lut = lut.repeat(3,3)
    dim = lut.shape[0]
    
    s = 3
    lut = lut[::s,::s,::s,:]
    # down_dim = lut.shape[0]
    x = y = z = np.arange(0,dim,3)
    x, y, z = np.meshgrid(x, y, z)
    lut = lut.reshape(-1, 3) # N,3
    lut = np.concatenate((lut, np.full((lut.shape[0],1), 0.12)), 1)
    ax.scatter(x, y, z, c=lut, s=point_size)  

def identity3d_tensor(dim): # 3,d,d,d
    step = np.arange(0,dim)/(dim-1) # Double, so need to specify dtype
    rgb = torch.tensor(step, dtype=torch.float32)
    LUT = torch.empty(3,dim,dim,dim)
    LUT[0] = rgb.unsqueeze(0).unsqueeze(0).expand(dim, dim, dim) # r
    LUT[1] = rgb.unsqueeze(-1).unsqueeze(0).expand(dim, dim, dim) # g
    LUT[2] = rgb.unsqueeze(-1).unsqueeze(-1).expand(dim, dim, dim) # b
    return LUT

if __name__ == "__main__":
    path = "a0038-MB_070908_135 22.8171.tif"
    dim = 33
    N = dim**3
    idt = identity3d_tensor(dim)
    # fullscreen()
    ax = plt.subplot(111, projection='3d')
    ax.set_xlabel('G', fontsize="xx-large")
    ax.set_ylabel('B', fontsize="xx-large")
    ax.set_zlabel("R", fontsize="xx-large")
    # ax.patch.set_facecolor((0,0,0,0))
    # ax.patch.set_alpha(0)
    ax.set_axis_off()
    
    neibors, proportion = count_one_image(path, dim)
    print(f'proportion={proportion*100} %')
    draw3DLUT(ax, idt)
    color = np.concatenate((neibors, np.ones((neibors.shape[0],1))),1)
    ax.scatter(neibors[:, 0], neibors[:, 1], neibors[:, 2], color=neibors/dim, s=30)
    ax.view_init(elev=13,azim=70)
    ax.set_box_aspect([1,1,0.95])
    plt.tight_layout()
    plt.show()
    
    