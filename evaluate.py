import os
from os.path import join 
import sys
import numpy as np
import pdb
import torch.nn as nn
from torchvision import utils
from torchvision.utils import save_image
from torchvision.utils import make_grid

from utils.losses import *
from parameter import *
from setting import *
import time

def evaluate(setting, epoch=None, best_psnr=None, do_save_img=True):
    eval_dataloader = setting.eval_dataloader
    opt = setting.opt
    if epoch is not None:
        epoch = "{:0>4}".format(epoch)
        dst = join(opt.save_images_root, epoch) 
    else:
        dst = opt.save_images_root

    os.makedirs(dst, exist_ok=True)  
    psnr_ls = []
    weight_ls = []
    psnr_in, psnr_out, avg_psnr_in, avg_psnr_out = 0, 0, 0, 0
    time_cost = [0, 0, 0]
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            targets = batch["target_org"].type(Tensor)
            imgs = batch["input_org"].type(Tensor)
            psnr_in = batch.get("psnr_in")
            if psnr_in is not None:
                psnr_in = psnr_in.squeeze()
            fakes, others = setting.evaluate(batch)
            for j, cost in enumerate(others['time_cost']):
                time_cost[j] += cost
            name = os.path.splitext(batch["name"][0])[0]
            if epoch is None:
                # sys.stdout.write("\r"+name+" "+" ".join([str(it) for it in others['time_cost']]))
                sys.stdout.write("\r"+name)
                # print(name)
            ########################################## log
            psnr_out = psnr(fakes, targets).item()
            if psnr_in is None:
                psnr_in = psnr(imgs, targets).item()
            change_str = "{:.4f}--{:.4f}".format(psnr_in, psnr_out)
            avg_psnr_in += psnr_in
            avg_psnr_out += psnr_out
            img_ls = [imgs.squeeze().data, fakes.squeeze().data, targets.squeeze().data]
            # save_image(img_ls, join(dst, "%s %s.jpg" % (name, change_str)), nrow=len(img_ls))

    isbest = ""
    avg_psnr_in /= len(eval_dataloader)
    avg_psnr_out /= len(eval_dataloader)
    if epoch is not None:
        if avg_psnr_out > best_psnr:
            isbest = "_best"
    change_str = "_%.4f--%.4f" % (avg_psnr_in, avg_psnr_out)
    os.rename(dst, dst + change_str + isbest) 
    # torch.cuda.empty_cache()
    for i in range(len(time_cost)):
        time_cost[i] /= len(eval_dataloader)
    # for i in range(len(time_cost)):
    #     time_cost[i] /= time_cost[-1]

    return avg_psnr_out, time_cost

if __name__ == "__main__":
    opt = parser.parse_args()
    setting = Setting(opt, "test")
    print(evaluate(setting))