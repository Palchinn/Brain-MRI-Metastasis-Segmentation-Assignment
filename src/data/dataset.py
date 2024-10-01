import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations import Compose, RandomRotate90, Flip, ElasticTransform

class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.mask_dir = os.path.join(self.root_dir, 'masks')
        self.image_files = sorted(os.listdir(self.image_dir))
        self.transform = transform
        
        if split == 'train':
            self.aug = Compose([
                RandomRotate90(),
                Flip(),
                ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5)
            ])
        else:
            self.aug = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.aug:
            augmented = self.aug(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        if self.transform:
            image = self.transform(image)
        
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        