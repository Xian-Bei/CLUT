import os
from os.path import join 
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from utils.LUT import *
from PIL import Image


def augment(img_input, img_target):
    try:
        W, H = img_input.size
    except:
        H,W = img_input.shape[1:]
    crop_h = round(H * np.random.uniform(0.6,1.))
    crop_w = round(W * np.random.uniform(0.6,1.))
    b = np.random.uniform(0.8,1.2)
    s = np.random.uniform(0.8,1.2)
    img_input = TF.adjust_brightness(img_input,b)
    img_input = TF.adjust_saturation(img_input,s)
    i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
    img_input = TF.resized_crop(img_input, i, j, h, w, (256, 256))
    img_target = TF.resized_crop(img_target, i, j, h, w, (256, 256))
    if np.random.random() > 0.5:
        img_input = TF.hflip(img_input)
        img_target = TF.hflip(img_target)
    if np.random.random() > 0.5:
        img_input = TF.vflip(img_input)
        img_target = TF.vflip(img_target)
    return img_input, img_target


class FiveK(Dataset):
    def __init__(self, data_root, split, model): 
        self.split = split
        self.model = model
        input_dir = join(data_root, "fiveK/input_"+split)
        target_dir = join(data_root, "fiveK/target_"+split)
        # input_dir = join(data_root, "fiveK/bit16_AsExpertC_Zero_OrgSize")
        # target_dir = join(data_root, "fiveK/bit8_expertC_OrgSize")
        input_files = sorted(os.listdir(input_dir))
        target_files = sorted(os.listdir(target_dir))
        self.input_files = [join(input_dir, file_name) for file_name in input_files]
        self.target_files = [join(target_dir, file_name) for file_name in target_files]


    def __getitem__(self, index):
        res = {}
        input_path = self.input_files[index]
        target_path = self.target_files[index]
        
        input_scale = 255
        target_scale = 255
        if 'CLUT' in self.model:
            img_input = TF.to_tensor(cv2.cvtColor(cv2.imread(input_path, -1), cv2.COLOR_BGR2RGB)/input_scale)
            img_target = TF.to_tensor(cv2.cvtColor(cv2.imread(target_path, -1), cv2.COLOR_BGR2RGB)/target_scale) 

            if self.split == "train":
                img_input, img_target = augment(img_input, img_target) 
                res["input"] = img_input.type(torch.FloatTensor)
                res["input_org"] = img_input.type(torch.FloatTensor)
                res["target"] = img_target.type(torch.FloatTensor)
                res["target_org"] = img_target.type(torch.FloatTensor)
            else:
                img_input_resize, img_target_resize = TF.resize(img_input, (256, 256)), TF.resize(img_target, (256, 256))
                res["input_org"] = img_input.type(torch.FloatTensor)
                res["target_org"] = img_target.type(torch.FloatTensor)
                res["input"] = img_input_resize.type(torch.FloatTensor)
                res["target"] = img_target_resize.type(torch.FloatTensor)

        elif 'Hash' in self.model:
            img_input = Image.open(input_path)
            img_target = Image.open(target_path)

            if self.split == "train":
                img_input, img_target = augment(img_input, img_target) # low-res
                res["input"] = TF.to_tensor(img_input)
                res["target"]= np.array(img_target, dtype=np.float32)/target_scale
            else:
                img_input_resize = TF.resize(img_input, (256, 256))
                res["input"] = TF.to_tensor(img_input_resize)
            res["input_org"] = np.array(img_input, dtype=np.float32)/input_scale
            res["target_org"] = np.array(img_target, dtype=np.float32)/target_scale
        else:
            raise 
            
        img_name = os.path.split(self.input_files[index])[-1]
        res["name"] = img_name
        return res 

    
    def __len__(self):
        return len(self.input_files)

# Implement your own DatasetClass according to your data format and dir arrangement.