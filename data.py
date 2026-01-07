from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
from .masks import get_mask

########################### INPAINTING SECTION

class InpaintDataset(Dataset):
    def __init__(self, root_dir, img_size=256, mask_type="irregular"):
        self.img_paths = []
        for ext in ("**/*.jpg","**/*.png","**/*.jpeg"):
            self.img_paths += list(Path(root_dir).glob(ext))
        self.img_paths = sorted(self.img_paths)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])
        self.mask_type = mask_type
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = str(self.img_paths[idx])
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        h,w = self.img_size, self.img_size
        mask_np = get_mask(self.mask_type, h, w)
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()  # 1,H,W
        masked_img = img * (1 - mask)
        return {"img": img, "masked_img": masked_img, "mask": mask, "path": p}

def make_dataloader(root, img_size, mask_type, batch_size, shuffle=True, num_workers=4, pin_memory_=True):
    ds = InpaintDataset(root, img_size=img_size, mask_type=mask_type)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory_)
