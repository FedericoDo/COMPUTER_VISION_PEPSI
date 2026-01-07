import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms.functional import gaussian_blur
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import os, yaml, random
from pathlib import Path
import numpy as np

######################### VARIOUS LOSSES 
def recon_loss(pred, target):
    return F.l1_loss(pred, target)
    
def structure_loss(s, target):
    target_blur = gaussian_blur(target, kernel_size=11)
    target_ds = F.interpolate(
        target_blur,
        size=s.shape[-2:],
        mode="bilinear",
        align_corners=False
    )
    target_ds = target_ds.mean(dim=1, keepdim=True)
    return F.l1_loss(s.mean(dim=1, keepdim=True), target_ds)

def texture_loss(pred, target, vgg):
    return vgg(pred, target)

class VGGLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        self.vgg = vgg
        self.layers = [3,8,15]  
        for p in self.vgg.parameters(): 
            p.requires_grad=False
        self.max_layer = max(self.layers)

    def forward(self, accessory, original):
        accessory_nor = (accessory - 0.5) * 2
        original_nor = (original - 0.5) * 2
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            accessory_nor = layer(accessory_nor)
            original_nor = layer(original_nor)
            if i in self.layers:
                loss += F.l1_loss(accessory_nor, original_nor)
            if i >= self.max_layer:
                break
        return loss

def adv_hinge_loss_discriminator(real_logits, fake_logits):
    loss_real = torch.mean(F.relu(1.0 - real_logits))
    loss_fake = torch.mean(F.relu(1.0 + fake_logits))
    return 0.5 * (loss_real + loss_fake)

def adv_hinge_loss_generator(fake_logits):
    return -torch.mean(fake_logits)


################### MASKS DEFINITION

def random_box_mask(h, w, min_frac=0.05, max_frac=0.5):
    mask = np.zeros((h,w), np.uint8)
    bh = int(random.uniform(min_frac, max_frac) * h)
    bw = int(random.uniform(min_frac, max_frac) * w)
    top = random.randint(0, h - bh)
    left = random.randint(0, w - bw)
    mask[top:top+bh, left:left+bw] = 1
    return mask

def random_irregular_mask(h, w, max_strokes=8, max_width=40):
    mask = np.zeros((h,w), np.uint8)
    for i in range(random.randint(1, max_strokes)):
        start_x = random.randint(0, w-1)
        start_y = random.randint(0, h-1)
        for o in range(random.randint(10, 60)):
            angle = random.random() * 2 * np.pi
            length = int(random.random() * max_width)
            dx = int(np.cos(angle) * length)
            dy = int(np.sin(angle) * length)
            end_x = np.clip(start_x + dx, 0, w-1)
            end_y = np.clip(start_y + dy, 0, h-1)
            thickness = random.randint(6, max(6, max_width//4))
            cv2.line(mask, (start_x,start_y), (end_x,end_y), 1, thickness) #0 è colore nero
            start_x, start_y = end_x, end_y
    return mask

def get_mask(mask_type, h, w):
    if mask_type == "bbox":
        return random_box_mask(h,w)
    if mask_type == "irregular":
        return random_irregular_mask(h,w)
    if random.random() < 0.5:
        return random_irregular_mask(h,w)
    else:
        return random_box_mask(h,w)


#################### METRICS METHODS

def batch_psnr(preds, targets):
    preds = preds.permute(0,2,3,1).cpu().numpy()
    targets = targets.permute(0,2,3,1).cpu().numpy()
    vals=[]
    for p,t in zip(preds,targets):
        p = np.clip(p,0,1); t = np.clip(t,0,1)
        vals.append(compare_psnr(t, p, data_range=1.0))
    return float(np.mean(vals))

def batch_ssim(preds, targets):
    preds = preds.permute(0,2,3,1).cpu().numpy()
    targets = targets.permute(0,2,3,1).cpu().numpy()
    vals=[]
    # faccio permute perché ssim, così come psnr si aspetta i canali sul terzo campo
    # il for scorre lungo il batch e quindi quello che estrae sono p e t del tipo [C, H, W]000
    for p,t in zip(preds,targets):
        p = np.clip(p,0,1); t = np.clip(t,0,1)
        vals.append(compare_ssim(t, p, channel_axis=2, data_range=1.0, win_size=11))
    return float(np.mean(vals))


####################### SOME GENERIC FUNCTIONS

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)