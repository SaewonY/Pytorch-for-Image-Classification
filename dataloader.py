import os
import numpy as np
import pathlib
import pandas as pd
import pickle as pkl
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from nsml import DATASET_PATH


class TestDataset(Dataset):
    def __init__(self, image_data_path, meta_data, transform=None):
        self.meta_data = meta_data
        self.image_dir = image_data_path
        self.transform = transform
        
    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir , str(self.meta_data['package_id'].iloc[idx]) , str(self.meta_data['sticker_id'].iloc[idx]) + '.png')
        png = Image.open(img_name).convert('RGBA')
        png.load() # required for png.split()

        new_img = Image.new("RGB", png.size, (255, 255, 255))
        new_img.paste(png, mask=png.split()[3]) # 3 is the alpha channel

        if self.transform:
            new_img = self.transform(new_img)
        
        return new_img


class AIRushDataset(Dataset):
    def __init__(self, df, image_data_path, transform=None):
        self.df = df
        self.image_dir = image_data_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir , str(self.df['package_id'].iloc[idx]) , str(self.df['sticker_id'].iloc[idx]) + '.png')
        png = Image.open(img_name).convert('RGBA')
        png.load() # required for png.split()

        new_img = Image.new("RGB", png.size, (255, 255, 255))
        new_img.paste(png, mask=png.split()[3]) # 3 is the alpha channel

        if self.transform:
            new_img = self.transform(new_img)
        
        tag = self.df['tag'][idx]
        
        return new_img, tag
        

def make_loader(df, image_dir, transforms, batch_size=256, num_workers=4):

    dataset = AIRushDataset(df, image_dir, transforms)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True, 
                        num_workers=num_workers, 
                        pin_memory=True)

    return loader



