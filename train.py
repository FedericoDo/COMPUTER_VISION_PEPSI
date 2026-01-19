# scripts/train.py
import os, argparse
import torch.nn.functional as F
import torch
from torch import optim
from torch.amp import autocast, GradScaler
from torchvision.utils import save_image
from tqdm import tqdm
from utils import *
from network import *
from globals import *
from data import *

def build_model(name, device):
    if name == "pepsi":
        return PEPSI(in_ch=4, base_ch=BASE_CH, num_res=NUM_RES).to(device)
    elif name == "pepsi_pp":  
        return PEPSIPlusPlus(base_ch=BASE_CH, num_asb=NUM_ASB).to(device)     
    elif name == "partial_conv":
        return PartialConvUNet(in_ch=4, base_ch=BASE_CH).to(device)
    elif name == "lama_like":
        return LaMaLike(in_ch=4, base_ch=BASE_CH).to(device)
    else:
        raise ValueError("Unknown model")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pepsi_pp")
    args = parser.parse_args()
    set_seed(SEED)
    model = args.model
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ensure_dir(CKPT_DIR)
    train_loader = make_dataloader(COCO_ROOT, IMG_SIZE, MASK_TYPE,
                                   BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = make_dataloader(PLACES2_ROOT, IMG_SIZE, MASK_TYPE,
                                 BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    G = build_model(model, device)
    D = PatchDiscriminator().to(device)
    vgg = VGGLoss(device=device)
    opt_g = optim.Adam(G.parameters(), lr=LR_G, betas=(B1,B2))
    opt_d = optim.Adam(D.parameters(), lr=LR_D, betas=(B1,B2))
    scaler = GradScaler('cuda',enabled=MIXED_PRECISION)
    global_step = 0
    for epoch in range(EPOCHS):
        global_step = 0
        G.train()
        D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            imgs = batch["img"].to(device)
            masks = batch["mask"].to(device)
            masked = batch["masked_img"].to(device)
            # Discriminator 
            with torch.no_grad():
                out = G(masked, masks)
                if model == "pepsi" or model == "pepsi_pp" :
                    comp_d = out[0]
                else:
                    comp_d = out
            with autocast('cuda',enabled=MIXED_PRECISION):
                fake_logits = D(comp_d.detach(), masks)
                real_logits = D(imgs, masks)
                l_d = adv_hinge_loss_discriminator(real_logits, fake_logits)
            opt_d.zero_grad()
            scaler.scale(l_d).backward()
            scaler.step(opt_d)
            scaler.update()
            # Generator 
            with autocast('cuda', enabled=MIXED_PRECISION):
                out_g = G(masked, masks)
                if model == "pepsi_pp":
                    comp, coarse = out_g
                    l_l1 = torch.mean(torch.abs(comp - imgs))
                    fake_logits_g = D(comp, masks)
                    l_adv = adv_hinge_loss_generator(fake_logits_g)
                    l_g = LOSS_L1 * l_l1 + LOSS_ADV * l_adv
                    l_c = torch.mean(torch.abs(coarse - imgs))
                    alpha = 1.0 - float(global_step) / float(EPOCHS*COARSE_FACTOR)
                    alpha = max(alpha, 0.0) 
                    l_total = l_g + LOSS_LS_MULT * alpha * l_c
                elif model == "pepsi":
                    comp, s, t = out_g
                    l_recon = recon_loss(comp, imgs)
                    l_perc = vgg(comp, imgs)
                    fake_logits_g = D(comp, masks)
                    l_g_adv = adv_hinge_loss_generator(fake_logits_g)
                    l_struct = structure_loss(s, imgs)
                    t_up = F.interpolate(
                        t,
                        size=imgs.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    l_text = vgg(t_up, imgs)
                    l_total = LOSS_RECON*l_recon + LOSS_PERCEPTUAL * l_perc + LOSS_ADV*l_g_adv + LOSS_STRUCT * l_struct + LOSS_TEXT * l_text
                else:
                    comp = out_g
                    l_recon = recon_loss(comp, imgs)
                    l_perc = vgg(comp, imgs)
                    fake_logits_g = D(comp, masks)
                    l_g_adv = adv_hinge_loss_generator(fake_logits_g)
                    l_total = LOSS_RECON*l_recon + LOSS_PERCEPTUAL * l_perc + LOSS_ADV*l_g_adv
            opt_g.zero_grad()
            scaler.scale(l_total).backward()
            scaler.step(opt_g)
            scaler.update()
            global_step += 1
            if global_step % 100 == 0:
                if (model == "pepsi_pp"):
                    pbar.set_postfix({"L_l1": l_l1.item(),"L_adv": l_adv.item(), "L_c": l_c.item(), "L_g": l_g.item()})
                elif (model == "pepsi"):
                    pbar.set_postfix({"L_recon": l_recon.item(),"L_perc": l_perc.item(), "L_adv": l_g_adv.item(), "L_struct": l_struct.item(), "L_text": l_text.item()})
                else:
                    pbar.set_postfix({"L_recon": l_recon.item(),"L_perc": l_perc.item(), "L_adv": l_g_adv.item()})
        # Validation + save
        if (epoch+1) % SAVE_EVERY == 0 or (epoch+1)==EPOCHS:
            G.eval()
            psrs=[]
            ssms=[]
            sample_dir = os.path.join(CKPT_DIR, "samples")
            ensure_dir(sample_dir)
            with torch.no_grad():
                for i, vb in enumerate(tqdm(val_loader, desc="Val")):
                    v_img = vb["img"].to(device)
                    v_mask = vb["mask"].to(device)
                    v_masked = vb["masked_img"].to(device)
                    out = G(v_masked, v_mask)
                    if model == "pepsi" or model == "pepsi_pp" :
                        comp = out[0]
                    else:
                        comp = out
                    psrs.append(batch_psnr(comp, v_img))
                    ssms.append(batch_ssim(comp, v_img))
                    if i < VALID_SAMP: 
                        grid = torch.cat([v_masked[:4], comp[:4], v_img[:4]], dim=0)
                        save_image(grid, os.path.join(sample_dir, f"model_{args.model}_epoch{epoch+1}_batch{i}.png"), nrow=4)
            avg_psnr = sum(psrs)/(len(psrs)+0.0001)
            avg_ssim = sum(ssms)/(len(ssms)+0.0001)
            print(f"Epoch {epoch+1} VAL PSNR: {avg_psnr:.4f} SSIM: {avg_ssim:.4f}")
            ckpt = {
                "epoch": epoch+1,
                "G_state": G.state_dict(),
                "D_state": D.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict()
            }
            torch.save(ckpt, os.path.join(CKPT_DIR, f"ckpt_{args.model}_epoch{epoch+1}.pth"))

if __name__ == "__main__":
    train()
