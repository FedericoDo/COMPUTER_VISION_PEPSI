import argparse, os
import torch
import torch.nn.functional as F
from utils import *
from network import *
from globals import *
from data import *
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image


def load_model_by_name(name, ckpt_path, device):
    if name == "pepsi":
        model = PEPSI(in_ch=4, base_ch=BASE_CH, num_res=NUM_RES).to(device)
    elif name == "pepsi_pp":  
        model = PEPSIPlusPlus(base_ch=BASE_CH, num_asb=NUM_ASB).to(device)     
    elif name == "lama_like":
        model = LaMaLike(in_ch=4, base_ch=BASE_CH).to(device)
    elif name == "partial_conv":
        model = PartialConvUNet(in_ch=4, base_ch=BASE_CH).to(device)
    else:
        raise ValueError(f"Unknown model name: {name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["G_state"])
    model.eval()
    return model

def evaluate(ckpt, model_name, dataset_root, out_dir):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = load_model_by_name(model_name, ckpt, device)
    loader = make_dataloader(dataset_root, IMG_SIZE, MASK_TYPE, BATCH_SIZE, False, NUM_WORKERS)
    ps = []
    ss = []
    set_seed(SEED)
    model_dir=out_dir+"_"+model_name
    compare_dir = os.path.join(model_dir, "compare")
    orig_dir = os.path.join(model_dir, "orig")
    genrated_dir = os.path.join(model_dir, "generated")
    ensure_dir(model_dir)
    ensure_dir(compare_dir)
    ensure_dir(orig_dir)
    ensure_dir(genrated_dir)
    with torch.no_grad():
        for i,b in enumerate(loader):
            img = b["img"].to(device)
            mask=b["mask"].to(device)
            masked=b["masked_img"].to(device)
            out = model(masked, mask)
            if model_name == "pepsi" or model_name == "pepsi_pp" :
                comp = out[0]
            else:
                comp = out
            ps.append(batch_psnr(comp, img))
            ss.append(batch_ssim(comp, img))
            for j in range(comp.size(0)):
                comp_img = comp[j]
                orig_path = b['path'][j] 
                masked_img = masked[j]
                orig_img = Image.open(orig_path).convert("RGB")
                orig_w, orig_h = orig_img.size               
                comp_resized = F.interpolate(comp_img.unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(0)
                mask_resized = F.interpolate(masked_img.unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(0)
                to_tensor = transforms.ToTensor()
                orig_tensor = to_tensor(orig_img).to(device)
                combined = torch.cat([orig_tensor, mask_resized, comp_resized],dim=2)
                path_compare = os.path.join(compare_dir, f"comp_{i*comp.size(0)+j}.png")
                path_orig = os.path.join(orig_dir, f"comp_{i*comp.size(0)+j}.png")
                path_gen = os.path.join(genrated_dir, f"comp_{i*comp.size(0)+j}.png")
                save_image(combined, path_compare)
                save_image(orig_tensor, path_orig)
                save_image(comp_resized, path_gen)
    print("PSNR:", sum(ps)/(len(ps)+0.0001), "SSIM:", sum(ss)/(len(ss)+0.0001))
    print("Saved generated images in", model_dir)
    print("To compute FID: pytorch-fid <path real images> <path generated>")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="pepsi_pp")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default="eval_out")
    args=p.parse_args()
    evaluate(args.ckpt, args.model, args.dataset, args.out)
