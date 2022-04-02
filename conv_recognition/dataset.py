import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torchvision.io import read_image

class UcoDataset(Dataset):

    def __init__(self, img_dir, sequence, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.sequence = sequence
        self.img_labels = []
        self.path = f'{img_dir}/sequence/'
        with open(os.path.join(img_dir,'annotations','annotations_frame', f'{sequence}.txt'), 'r') as f:
            lines = f.readlines()
            self.img_labels = [line.split(' ')[1] for line in lines]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 'frames', self.sequence, f'{idx:06d}.jpg')
        image = read_image(img_path)
        label = self.img_labels[idx]
        return image, label
        
        