import os
import shutil
import sys
import math
import numpy as np
import pdb
import time

import torch.nn as nn

from parameter import *
from utils.losses import *
from setting import *
from evaluate import evaluate


def train(setting):
    optimizer = setting.optimizer
    train_dataloader = setting.train_dataloader
    opt = setting.opt
    best_psnr = 0
    best_epoch = 0
    for epoch in range(opt.epoch, opt.n_epochs+1):
        avg_img_loss_ls = [0 for loss_name in opt.losses]
        avg_other_loss = 0 
        avg_psnr = 0
        log_template = "\rE {:0>3d}/".format(epoch) + str(opt.n_epochs) + " B {:0>4d}/" + str(len(train_dataloader))
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            fakes, others = setting.train(batch)
            targets = batch["target"].type(Tensor)
            log_str = log_template.format(i+1)
            loss = 0
            for loss_idx, loss_name in enumerate(opt.losses):
                weight, loss_name = loss_name.split("*")
                img_loss = eval(loss_name)(fakes, targets, float(weight))
                loss += img_loss
                avg_img_loss_ls[loss_idx] += img_loss.item()
            other_loss = others.get("other_loss")
            if isinstance(other_loss, torch.Tensor):
                loss += other_loss
                avg_other_loss += other_loss.item()
            loss.backward()
            optimizer.step()
            ######################################### Log
            avg_psnr += psnr(fakes, targets).item()
            log_str += " psnr:{:.4f}".format(avg_psnr/(i+1))
            for loss_idx, loss_name in enumerate(opt.losses):
                log_str += " " + loss_name + ":%.5f"%(avg_img_loss_ls[loss_idx]/(i+1))
            if isinstance(other_loss, torch.Tensor):
                log_str += " other_loss:{:.5f}".format(avg_other_loss/(i+1))
            sys.stdout.write(log_str)

        ################################ eval and save
        setting.save_ckp(None, True) # model+optimizer
        if epoch % opt.checkpoint_interval == 0:
            eval_psnr = evaluate(setting, epoch, best_psnr) 
            if eval_psnr > best_psnr:
                best_psnr = eval_psnr
                best_epoch = epoch
                setting.save_ckp(epoch, False)
            log_str += " best_psnr:{:.4f} ".format(best_psnr)
            log_str += "best_epoch:%3d" % (best_epoch)
            with open(opt.save_logs_root+".txt", "a") as f: # save log
                f.write(log_str)
            sys.stdout.write(" best_psnr:{:.4f} best_epoch:{:3d} \n".format(best_psnr, best_epoch))
        

if __name__ == "__main__":
    opt = parser.parse_args()
    setting = Setting(opt)
    train(setting)
