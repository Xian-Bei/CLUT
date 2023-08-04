import os
from os.path import join
import numpy as np
from thop import profile
from models import *
from parameters import cuda, Tensor, device
from torch.utils.data import DataLoader
from datasets import *

class Setting():
    
    def __init__(self, hparams, mode="train"):

        
        else: # python eval.py
            self.epoch = hparams.epoch
            hparams.save_images_root = hparams.output_dir +"_"+ str(self.epoch)
            if hparams.epoch > 1:
                load = torch.load(join(hparams.save_models_root, "model{:0>4}.pth".format(hparams.epoch)))
                self.model.load_state_dict(load, strict=True)
                print("model loaded from epoch "+str(hparams.epoch))
        
        self.
        os.makedirs(hparams.save_images_root, exist_ok=True)
        
             
    def train(self, batch):
        
        return fakes, others

    def evaluate(self, batch):
        self.model.eval()
        img = batch["input"].type(Tensor)
        img_org = batch.get("input_org").type(Tensor)
        fake, others = self.model(img, img_org)

        return fake, others

    def save_ckp(self, epoch=None, save_opt=True):
        if epoch is not None:
            torch.save(self.model.state_dict(), "{}/model{:0>4}.pth".format(self.hparams.save_models_root, epoch))
        else:
            torch.save(self.model.state_dict(), "{}/model_latest.pth".format(self.hparams.save_models_root))
        if save_opt:
            torch.save(self.optimizer.state_dict(), "{}/optimizer_latest.pth".format(self.hparams.save_models_root))
   
