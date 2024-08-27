import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register


@register('image-folder')
class ImageFolder(Dataset): #处理单个文件夹内的图像数据集加载和预处理
    def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', mask=False,dep=False):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key

        self.size = size
        self.mask = mask
        self.dep = dep
        if self.mask: #GT的mask图像
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=Image.NEAREST), #缩放
                transforms.ToTensor(),
            ])

        elif self.dep: #0820深度

            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.38], std=[0.35])
            ])
            
        else: #RGB图像
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) #对图像标准化处理
            ])

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []

        for filename in filenames:
            file = os.path.join(path, filename)
            self.append_file(file)

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return self.img_process(x)
        elif self.cache == 'in_memory':
            return x

    def img_process(self, file):
        if self.mask:
            return Image.open(file).convert('L')
        elif self.dep:
            return Image.open(file).convert('L')
        else:
            return Image.open(file).convert('RGB')

@register('paired-image-folders')
class PairedImageFolders(Dataset):# 处理三个文件夹内的成对图像数据集加载

    def __init__(self, root_path_1, root_path_2, root_path_3, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs, mask=True)
        self.dataset_3 = ImageFolder(root_path_3, **kwargs, dep=True) # 820深度图

    def __len__(self):
        return len(self.dataset_1)

    # def __getitem__(self, idx):
    #     return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx]

    def __getitem__(self, idx):
        img = self.dataset_1[idx]
        mask = self.dataset_2[idx]
        dep = self.dataset_3[idx]
        
        # # 打印每个索引处的文件名
        # print(f'Index {idx}:')
        # print(f'  Image Path: {self.dataset_1.files[idx % len(self.dataset_1.files)]}')
        # print(f'  Mask Path: {self.dataset_2.files[idx % len(self.dataset_2.files)]}')
        # print(f'  Depth Path: {self.dataset_3.files[idx % len(self.dataset_3.files)]}')
    

        return img, mask, dep
