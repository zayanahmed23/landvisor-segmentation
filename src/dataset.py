import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class LoveDASparseDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_points=10, ignore_index=-1, transform=False, mode='train'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = sorted([
            f for f in os.listdir(img_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ])
        
        self.num_points = num_points
        self.ignore_index = ignore_index
        self.transform = transform
        self.mode = mode
        
        # LoveDA classes are 1-7, but PyTorch's CrossEntropyLoss requires 0-indexed contiguous classes.
        self.class_shift = -1 

    def __len__(self):
        return len(self.img_names)

    def _generate_point_mask(self, full_mask):
        # TODO: Pre-compute these sparse masks offline to speed up the dataloader bottleneck
        sp_mask = np.full(full_mask.shape, self.ignore_index, dtype=np.int64)
        unique_classes = np.unique(full_mask)
        
        for cls in unique_classes:
            if cls == 0: continue 
            
            y_coords, x_coords = np.where(full_mask == cls)
            if len(y_coords) > 0:
                n_to_sample = min(self.num_points, len(y_coords))
                indices = random.sample(range(len(y_coords)), n_to_sample)
                
                sel_y = y_coords[indices]
                sel_x = x_coords[indices]
                
                sp_mask[sel_y, sel_x] = cls + self.class_shift
                
        return sp_mask

    def transform_data(self, img, sp_mask, gt_tensor):
        # Standard torchvision transforms don't apply identical random states to multiple tensors.
        # We manually apply identical flips here to ensure the sparse mask and image remain perfectly spatially aligned.
        if random.random() > 0.5:
            img = TF.hflip(img)
            sp_mask = TF.hflip(sp_mask)
            gt_tensor = TF.hflip(gt_tensor)
        if random.random() > 0.5:
            img = TF.vflip(img)
            sp_mask = TF.vflip(sp_mask)
            gt_tensor = TF.vflip(gt_tensor)
        return img, sp_mask, gt_tensor

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base_name + '.png')
        
        # HARDWARE OPTIMIZATION: Downsampling native 1024x1024 imagery to 512x512. 
        # This prevents OOM (Out of Memory) errors on local CPU and enables rapid prototyping. 
        # (Note: Use Image.NEAREST for masks to prevent generating fake fractional class labels during interpolation).

        img_pil = Image.open(img_path).convert("RGB").resize((512, 512), Image.BILINEAR)
        mask_pil = Image.open(mask_path).resize((512, 512), Image.NEAREST)
        
        full_mask_np = np.array(mask_pil, dtype=np.int64)
        sparse_mask_np = self._generate_point_mask(full_mask_np)
        
        img_tsr = TF.to_tensor(img_pil) 
        sp_mask_tsr = torch.from_numpy(sparse_mask_np).long()
        gt_tsr = torch.from_numpy(full_mask_np + self.class_shift).long()
        
        if self.mode == 'train' and self.transform:
            img_tsr, sp_mask_tsr, gt_tsr = self.transform_data(img_tsr, sp_mask_tsr, gt_tsr)

        return img_tsr, sp_mask_tsr, gt_tsr