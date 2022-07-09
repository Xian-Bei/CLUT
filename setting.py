import os
from os.path import join
import numpy as np
from thop import profile
from models import *
from parameter import cuda, Tensor, device
from torch.utils.data import DataLoader
from datasets import *

class Setting():
    
    def __init__(self, opt, mode="train"):

        self.opt = opt
        opt.losses = opt.losses.split(" ")
        opt.start_eval_epoch = 1

        self.model = CLUTNet(opt.model, dim=opt.dim)
        self.model = self.model.to(device)
        opt.output_dir = join(opt.dataset, opt.model)
        opt.save_models_root = join(opt.save_root, opt.output_dir +"_"+ "models")
        self.eval_dataloader = DataLoader(
            eval(opt.dataset)(opt.data_root, mode="test"),
            batch_size=1,
            shuffle=False,
            num_workers=opt.n_cpu,
        )

        if mode == "train": # python train.py
            os.makedirs(opt.save_models_root, exist_ok=True)
            opt.save_logs_root = join(opt.save_root, "logs", opt.dataset) 
            os.makedirs(opt.save_logs_root, exist_ok=True)
            opt.save_images_root = join(opt.save_root, opt.output_dir +"_"+ "images")
            os.makedirs(opt.save_images_root, exist_ok=True)
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=opt.lr,
            )
            self.train_dataloader = DataLoader(
                eval(opt.dataset)(opt.data_root, mode="train"),
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_cpu,
            )
            ###################################### load ckpt
            if opt.epoch > 1:
                self.optimizer.load_state_dict(torch.load(join(opt.save_models_root, "optimizer_latest.pth")))
                if os.path.exists(join(opt.save_models_root, "model{:0>4}.pth".format(opt.epoch-1))):
                    self.model.load_state_dict(torch.load(join(opt.save_models_root, "model{:0>4}.pth".format(opt.epoch-1))), strict=True)
                    print("ckp loaded from epoch " + str(opt.epoch-1))
                else:
                    self.model.load_state_dict(torch.load(join(opt.save_models_root, "model_latest.pth")), strict=True)
                    print("ckp loaded from the latest epoch")
        else: # python eval.py
            self.epoch = opt.epoch
            opt.save_images_root = join(opt.save_root, opt.output_dir +"_"+ str(self.epoch))
            if opt.epoch > 1:
                load = torch.load(join(opt.save_models_root, "model{:0>4}.pth".format(opt.epoch)))
                self.model.load_state_dict(load, strict=True)
                print("model loaded from epoch "+str(opt.epoch))
        
        self.TVMN = TVMN(opt.dim, mode="old").to(device)
        os.makedirs(opt.save_images_root, exist_ok=True)
        
             
    def train(self, batch):
        self.model.train()
        self.model.my_train()
        imgs = batch["input"].type(Tensor)
        experts = batch["target"].type(Tensor)
        # flops, params = profile(self.model, inputs = (imgs, imgs, self.TVMN))
        fakes, others = self.model(imgs, imgs, TVMN=self.TVMN)
        tvmn = others.get("tvmn")
        others["other_loss"] = self.opt.lambda_smooth*(tvmn[0]+10*tvmn[2]) + self.opt.lambda_mn*tvmn[1]
        
        return fakes, others

    def evaluate(self, batch):
        self.model.eval()
        img = batch["input"].type(Tensor)
        img_org = batch.get("input_org").type(Tensor)
        fake, others = self.model(img, img_org)

        return fake, others

    def save_ckp(self, epoch=None, save_opt=True):
        if epoch is not None:
            torch.save(self.model.state_dict(), "{}/model{:0>4}.pth".format(self.opt.save_models_root, epoch))
        else:
            torch.save(self.model.state_dict(), "{}/model_latest.pth".format(self.opt.save_models_root))
        if save_opt:
            torch.save(self.optimizer.state_dict(), "{}/optimizer_latest.pth".format(self.opt.save_models_root))
   
