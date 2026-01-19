import torch.nn as nn
import torch
import torch.nn.functional as F
from globals import *


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch=4
        base=64
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base*2), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*2, base*4, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(base*4), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*4, out_channels=1, kernel_size=4, padding=1)
        )
    def forward(self, img, mask):
        masked = torch.cat([img, mask], dim=1)
        return self.net(masked)


########################################### LAMA LIKE

class LaMaLike(nn.Module):
    def __init__(self, in_ch, base_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch*2, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*2, base_ch*4, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_channels=3, kernel_size=7, padding=3), nn.Sigmoid()
        )

    def forward(self, img, mask):
        masked = torch.cat([img, mask], dim=1)
        out = self.net(masked)
        comp = out * mask + img * (1 - mask)
        return comp


####################################### PARTIAL CONV LIKE

class PartialConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super().__init__()
        self.input_conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=True)

    def forward(self, x, mask):
        x_masked = x * mask
        with torch.no_grad():
            kernel = torch.ones((1,1,self.input_conv.kernel_size[0], self.input_conv.kernel_size[1]),
                                device=x.device)
            mask_sum = F.conv2d(mask, kernel, bias=None, stride=self.input_conv.stride, padding=self.input_conv.padding)
        raw = self.input_conv(x_masked)
        mask_sum_clamped = mask_sum.clone()
        mask_sum_clamped[mask_sum_clamped == 0] = 1.0
        out = raw / mask_sum_clamped
        new_mask = (mask_sum > 0).float()
        return out, new_mask

class PConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=1):
        super().__init__()
        self.pconv = PartialConv2d(in_ch, out_ch, kernel=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        # inplace = True serve per ridurre uso di memoria lavorando
        # direttamente su tensoere in ingresso
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        out, new_mask = self.pconv(x, mask)
        out = self.bn(out)
        out = self.activation(out)
        return out, new_mask

class PartialConvUNet(nn.Module):
    def __init__(self, in_ch, base_ch):
        super().__init__()
        # Encoder
        self.enc1 = PConvBlock(in_ch, base_ch, kernel=7, padding=3)
        self.enc2 = PConvBlock(base_ch, base_ch*2, kernel=5, stride=2, padding=2)
        self.enc3 = PConvBlock(base_ch*2, base_ch*4, kernel=5, stride=2, padding=2)
        # Bottleneck
        self.bottleneck = PConvBlock(base_ch*4, base_ch*4, kernel=3, padding=1)
        # Decoder 
        self.dec3 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(base_ch, out_channels=3, kernel_size=3, padding=1)
        # small fusions
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(base_ch*2)
        self.bn1 = nn.BatchNorm2d(base_ch)

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)  
        encoded1, mask1 = self.enc1(x, mask)        
        encoded2, mask2 = self.enc2(encoded1, mask1)
        encoded3, mask3 = self.enc3(encoded2, mask2)
        bottle, maskb = self.bottleneck(encoded3, mask3)
        decoded3 = self.dec3(bottle)            
        decoded3 = self.relu(self.bn2(decoded3))
        decoded3 = decoded3 + encoded2
        decoded2 = self.dec2(decoded3)
        decoded2 = self.relu(self.bn1(decoded2))
        decoded2 = decoded2 + encoded1
        out = torch.sigmoid(self.dec1(decoded2))
        comp = out * mask + img * (1 - mask)
        return comp

############################## PEPSI

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x): 
        return self.net(x)

class DepthwiseSepConv(nn.Module):
    def __init__(self, ch, kernel_size=3, padding=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size, padding=padding, groups=ch),
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x): 
        return self.op(x)
        
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSepConv(ch),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            DepthwiseSepConv(ch),
            nn.BatchNorm2d(ch)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(x + self.conv(x))

class PEPSI(nn.Module):
    def __init__(self, in_ch, base_ch, num_res):
        super().__init__()
        # initial conv
        self.head = ConvBlock(in_ch, base_ch, kernel_size=7, padding=3)
        # encoder
        self.enc1 = ConvBlock(base_ch, base_ch*2, kernel_size=4, stride=2, padding=1)
        self.enc2 = ConvBlock(base_ch*2, base_ch*4, kernel_size=4, stride=2, padding=1)
        # structure branch 
        self.struc_res = nn.Sequential(*[ResidualBlock(base_ch*4) for _ in range(num_res//2)])
        # texture branch 
        self.text_res = nn.Sequential(*[ResidualBlock(base_ch*4) for _ in range(num_res//2)])
        # fusion
        self.fuse = ConvBlock(base_ch*8, base_ch*4, kernel_size=3, padding=1)
        # decoder
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, 3, 7, padding=3),
             nn.Sigmoid()
        )
        self.struct_proj = nn.Conv2d(base_ch*4, 1, kernel_size=1)
        self.text_proj = nn.Conv2d(base_ch*4, 3, kernel_size=1)


    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)  
        h = self.head(x)
        e1 = self.enc1(h)
        e2 = self.enc2(e1)
        s = self.struc_res(e2)
        t = self.text_res(e2)
        fused = torch.cat([s, t], dim=1)
        fused = self.fuse(fused)
        d2 = self.dec2(fused) + e1  
        d1 = self.dec1(d2) + h
        out = self.out_conv(d1)
        comp = out * mask + img * (1 - mask)
        s_out = self.struct_proj(s)            
        t_out = self.text_proj(t)               
        t_out = torch.sigmoid(t_out)
        return comp, s_out, t_out




######################################### PEPSIPP


class ConvBlockPP(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlockPP(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))



class AdaptiveSharedBlock(nn.Module):
    def __init__(self, ch, gamma_mul, beta_mult):
        super().__init__()
        self.res = ResidualBlockPP(ch)
        self.gamma = nn.Parameter(torch.ones(1, ch, 1, 1) * gamma_mul).to(DEVICE)
        self.beta = nn.Parameter(torch.zeros(1, ch, 1, 1)* beta_mult).to(DEVICE)

    def forward(self, x):
        out = self.res(x)
        return self.gamma * out + self.beta


class CoarseGenerator(nn.Module):
    def __init__(self, in_ch=4, base_ch=64, num_asb=6):
        super().__init__()
        self.head = ConvBlockPP(in_ch, base_ch, k=7, p=3)

        self.enc1 = ConvBlockPP(base_ch, base_ch*2, k=4, s=2, p=1)
        self.enc2 = ConvBlockPP(base_ch*2, base_ch*4, k=4, s=2, p=1)

        self.asb = nn.Sequential(
            *[AdaptiveSharedBlock(base_ch*4, COARSE_GAMMA, COARSE_BETA) for _ in range(num_asb)]
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(base_ch, 3, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        h = self.head(x)
        e1 = self.enc1(h)
        e2 = self.enc2(e1)

        latent = self.asb(e2)

        d2 = self.dec2(latent) + e1
        d1 = self.dec1(d2) + h

        return self.out(d1)


class RefinementGenerator(nn.Module):
    def __init__(self, in_ch=7, base_ch=64, num_asb=6):
        super().__init__()
        self.head = ConvBlockPP(in_ch, base_ch, k=7, p=3)

        self.enc1 = ConvBlockPP(base_ch, base_ch*2, 4, 2, 1)
        self.enc2 = ConvBlockPP(base_ch*2, base_ch*4, 4, 2, 1)

        self.asb = nn.Sequential(
            *[AdaptiveSharedBlock(base_ch*4, REFINE_GAMMA, REFINE_BETA) for _ in range(num_asb)]
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(base_ch, 3, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, coarse, img, mask):
        x = torch.cat([coarse, img, mask], dim=1)
        h = self.head(x)
        e1 = self.enc1(h)
        e2 = self.enc2(e1)

        latent = self.asb(e2)

        d2 = self.dec2(latent) + e1
        d1 = self.dec1(d2) + h

        return self.out(d1)


class PEPSIPlusPlus(nn.Module):
    def __init__(self, base_ch=64, num_asb=6):
        super().__init__()
        self.coarse = CoarseGenerator(4, base_ch, num_asb)
        self.refine = RefinementGenerator(7, base_ch, num_asb)

    def forward(self, img, mask):
        coarse = self.coarse(img, mask)
        refined = self.refine(coarse, img, mask)
        comp = refined * mask + img * (1 - mask)
        return comp, coarse