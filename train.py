from parameters import *
import sys
from os.path import join 
import torch.nn as nn
from utils.losses import *
from test import test
from models import *
from datasets import *
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
from ipdb import set_trace as S

if __name__ == "__main__":
    hparams = parser.parse_args()
    if hparams.name is None:
        hparams.name = "_".join(hparams.model)
    if 'Hash' in hparams.model[0]:
        hparams.lr = 0.0005
    hparams.output_dir = join(hparams.save_root, hparams.dataset, hparams.name)
    os.makedirs(hparams.output_dir, exist_ok=True)
    print(f"ckpt will be saved to {hparams.output_dir}")
    hparams.save_models_root = hparams.output_dir
    hparams.save_logs_root = hparams.output_dir
    hparams.save_images_root = hparams.output_dir

    model = eval(hparams.model[0])(*hparams.model[1:]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.lr,
    )
    if 'Hash' in hparams.model[0]:
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=hparams.lr/10, verbose=True)
    else:
        scheduler = None
    train_dataloader = DataLoader(
        eval(hparams.dataset)(hparams.data_root, split="train", model=hparams.model[0]),
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
    )
    test_dataloader = DataLoader(
        eval(hparams.dataset)(hparams.data_root, split="test", model=hparams.model[0]),
        batch_size=1,
        shuffle=False,
        num_workers=hparams.num_workers,
    )
    if hparams.tvmn:
        TVMN = TVMN(hparams.model[-1], lambda_smooth=0.0001, lambda_mn=10.0).to(device)
    else:
        TVMN = None
    if hparams.epoch > 1:
        latest_ckpt = torch.load(join(hparams.save_models_root, "latest_ckpt.pth"))
        optimizer.load_state_dict(latest_ckpt['optimizer'])
        best_psnr = latest_ckpt['best_psnr']
        best_epoch = latest_ckpt['best_epoch']
        if scheduler:
            scheduler.load_state_dict(latest_ckpt['scheduler'])
        try:
            model.load_state_dict(torch.load(join(hparams.save_models_root, f"model_{hparams.epoch-1}.pth")), strict=True)
            sys.stdout.write(f"Successfully loading from {hparams.epoch-1} epoch ckpt\n")
        except:
            model.load_state_dict(latest_ckpt['model'], strict=True)
            sys.stdout.write(f"Successfully loading from the latest ckpt\n")
    else:
        best_psnr = 0
        best_epoch = 0
    N = len(train_dataloader)
    interval = N//50
    for epoch in range(hparams.epoch, hparams.num_epochs+1):
        model.train()
        loss_ls = [0 for loss in hparams.losses] + [0]
        epoch_start_time = time()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            inputs = batch["input"].to(device)
            inputs_org = batch.get("input_org").to(device)
            targets = batch["target"].to(device)
            # flops, params = profile(model, inputs = (inputs, inputs_org, self.TVMN))
            results = model(inputs, inputs_org, TVMN=TVMN)
            fakes = results["fakes"]
            loss_ls[-1] = results.get("tvmn_loss", 0)
            
            
            for loss_idx, loss_name in enumerate(hparams.losses):
                loss_ls[loss_idx] = eval(loss_name)(fakes, targets)
            sum(loss_ls).backward()
            optimizer.step()
            
            if i % interval == 0 or i == N-1:
                psnr_result = psnr(fakes, targets).item()
                log_train = f"\rE {epoch:>3d}/{hparams.num_epochs:>3d} B {i+1:>4d} PSNR:{psnr_result:>0.2f}dB "
                for loss_idx, loss_name in enumerate(hparams.losses):
                    log_train += f"{loss_name}:{loss_ls[loss_idx].item():>0.3f} "
                if isinstance(loss_ls[-1], torch.Tensor):
                    log_train += f"tvmn: {loss_ls[-1].item():>0.3f} "
                torch.cuda.synchronize()
                cost_time = (time() - epoch_start_time)/(i+1)
                left_time = cost_time*(N-(i+1))/60
                sys.stdout.write(log_train + f"left={left_time:0>4.2f}m ")
        
        torch.cuda.synchronize()
        cost_time = time() - epoch_start_time
        log_test = " epoch:{:.1f}s ".format(cost_time)

        eval_psnr, test_cost = test(model, test_dataloader, join(hparams.save_images_root, f"{epoch:0>4}"), best_psnr) 
        if eval_psnr > best_psnr:
            best_psnr = eval_psnr
            best_epoch = epoch
            torch.save(model.state_dict(), f"{hparams.save_models_root}/model{epoch:0>4}.pth")

        log_test += f"Test:{eval_psnr:>0.2f}dB {test_cost:0>5.2f}s best:{best_psnr:.2f}dB {best_epoch:3d}. "
        # sys.stdout.write(log_test)
        print(log_test)
        with open(join(hparams.save_logs_root, "log.txt"), "a") as f: # save log
            f.write(log_train + log_test)
        
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr,
            'best_epoch': best_epoch,
        }
        if scheduler is not None:
            scheduler.step()
            ckpt['scheduler'] = scheduler.state_dict()
            
        torch.save(ckpt, f"{hparams.save_models_root}/latest_ckpt.pth")