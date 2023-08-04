from parameters import *
import sys
from os.path import join
import torch.nn as nn
from torchvision.utils import save_image
from utils.losses import *
from models import *
from datasets import *
from torch.utils.data import DataLoader 
from ipdb import set_trace as S

@torch.no_grad()
def test(model, test_dataloader, save_path, best_psnr=None, save_img=False):
    model.eval()
    os.makedirs(save_path, exist_ok=True)  
    avg_psnr_out = 0
    test_start = time()
    for i, batch in enumerate(test_dataloader):
        inputs = batch["input"].to(device)
        inputs_org = batch.get("input_org").to(device)
        targets_org = batch["target_org"].to(device)
        name = os.path.splitext(batch["name"][0])[0]
        # flops, params = profile(self.model, inputs = (imgs, imgs, self.TVMN))
        results = model(inputs, inputs_org, TVMN=None)
        fakes = results["fakes"]
        
        psnr_out = psnr(fakes, targets_org).item()
        avg_psnr_out += psnr_out
        if save_img:
            img_ls = [inputs_org.squeeze().data, fakes.squeeze().data, targets_org.squeeze().data]
            if img_ls[0].shape[0] > 3:
                img_ls = [img.permute(2,0,1) for img in img_ls]
            save_image(img_ls, join(save_path, f"{name}_{psnr_out:.2f}.jpg"), nrow=len(img_ls))
        # sys.stdout.write(f"\r{name} {psnr_out:.2f}dB")
        
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    test_cost = time() - test_start

    avg_psnr_out /= len(test_dataloader)
    new_folder_name = save_path + f" {avg_psnr_out:.2f}dB {test_cost:0>5.2f}s"
    if best_psnr is not None and avg_psnr_out > best_psnr:
        new_folder_name += '_best'
    os.rename(save_path,  new_folder_name) 

    return avg_psnr_out, test_cost

if __name__ == "__main__":
    hparams = parser.parse_args()
    model = eval(hparams.model[0])(*hparams.model[1:]).to(device)
    if hparams.name is None:
        hparams.name = "_".join(hparams.model)
    hparams.output_dir = join(hparams.save_root, hparams.dataset, hparams.name)
    hparams.save_models_root = hparams.output_dir
    hparams.save_images_root = join(hparams.output_dir, f"{hparams.epoch:0>4}_test")

    ckpt = torch.load(join(hparams.save_models_root, f"model{hparams.epoch:0>4}.pth"))
    model.load_state_dict(ckpt, strict=True)
    sys.stdout.write(f"Successfully loading from epoch {hparams.epoch}\n")
    test_dataloader = DataLoader(
        eval(hparams.dataset)(hparams.data_root, split="test", model=hparams.model[0]),
        batch_size=1,
        shuffle=False,
        num_workers=hparams.num_workers,
    )
    print("\rTest results: {:>0.2f}dB; Test cost: {:0>5.2f}s".format(*test(model, test_dataloader, hparams.save_images_root, save_img=True)))