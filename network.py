import torch.nn as nn
import torch
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch=4
        base=64
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel=4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base*2, kernel=4, stride=2, padding=1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*2, base*4, kernel=4, stride=2, padding=1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*4, out_ch=1, kernel=4, padding=1)
        )
    def forward(self, img, mask):
        masked = torch.cat([img, mask], dim=1)
        return self.net(masked)


########################################### LAMA LIKE

class LaMaLike(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch=4
        base_ch=64
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch*2, kernel=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*2, base_ch*4, kernel=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch*2, base_ch, kernel=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch=3, kernel=7, padding=3), nn.Sigmoid()
        )

    def forward(self, img, mask):
        masked = torch.cat([img, mask], dim=1)
        out = self.net(masked)
        comp = out * mask + img * (1 - mask)
        return comp


####################################### PARTIAL CONV LIKE

class PartialConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
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
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
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
    def __init__(self, in_ch=4, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = PConvBlock(in_ch, base_ch, kernel=7, padding=3)
        self.enc2 = PConvBlock(base_ch, base_ch*2, kernel=5, stride=2, padding=2)
        self.enc3 = PConvBlock(base_ch*2, base_ch*4, kernel=5, stride=2, padding=2)
        # Bottleneck
        self.bottleneck = PConvBlock(base_ch*4, base_ch*4, kernel=3, padding=1)
        # Decoder 
        self.dec3 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel=3, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel=3, stride=2, padding=1)
        self.dec1 = nn.Conv2d(base_ch, out_ch=3, kernel=3, padding=1)
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

############################## PEPSI++

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
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
            nn.Conv2d(ch, ch, kernel_size, padding=padding, groups=in_ch),
            nn.Conv2d(ch, ch, kernel=1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x): 
        return self.op(x)
        
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSepConv(ch, ch),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            DepthwiseSepConv(ch, ch),
            nn.BatchNorm2d(ch)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(x + self.conv(x))

class PEPSIPlusPlus(nn.Module):
    def __init__(self, in_ch=4, base_ch=64, num_res=8):
        super().__init__()
        # initial conv
        self.head = ConvBlock(in_ch, base_ch, k=7, p=3)
        # encoder (shared)
        self.enc1 = ConvBlock(base_ch, base_ch*2, k=3, s=2, p=1)
        self.enc2 = ConvBlock(base_ch*2, base_ch*4, k=3, s=2, p=1)
        # structure branch (larger RF)
        self.struc_res = nn.Sequential(*[ResidualBlock(base_ch*4) for _ in range(num_res//2)])
        # texture branch (more shallow)
        self.text_res = nn.Sequential(*[ResidualBlock(base_ch*4) for _ in range(num_res//2)])
        # fusion
        self.fuse = ConvBlock(base_ch*8, base_ch*4, k=3, p=1)
        # decoder
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2, base_ch, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(nn.Conv2d(base_ch, 3, 5, padding=2), nn.Sigmoid())
        self.struct_head = nn.Conv2d(base_ch*4, 3, kernel_size=1)
        self.text_head   = nn.Conv2d(base_ch*4, 3, kernel_size=1)


    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)  # B,4,H,W
        h = self.head(x)
        e1 = self.enc1(h)
        e2 = self.enc2(e1)
        s = self.struc_res(e2)
        t = self.text_res(e2)
        fused = torch.cat([s, t], dim=1)
        fused = self.fuse(fused)
        d2 = self.dec2(fused) + e1  # skip
        d1 = self.dec1(d2) + h
        out = self.out_conv(d1)
        comp = out * mask + img * (1 - mask)
        s_img = F.interpolate(self.struct_head(s), size=img.shape[-2:], mode="bilinear", align_corners=False)
        t_img = F.interpolate(self.text_head(t), size=img.shape[-2:], mode="bilinear", align_corners=False)
        return comp, s_img, t_img
