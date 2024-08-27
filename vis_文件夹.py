import os
import sys
import cv2
import torch
import numpy as np
# import lightning as L
import matplotlib.pyplot as plt
# from lightning.fabric.fabric import _FabricOptimizer
# from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

from PIL import Image

import argparse
import os
import glob

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
# from mmcv.runner import load_checkpoint

class DepthProcessor:
    def __init__(self, inp_size):
        self.inp_size = inp_size
        self.dep_transform = transforms.Compose([
            transforms.Resize((self.inp_size, self.inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.38], std=[0.35])
        ])

    def process_depth(self, depth_image):
        # 将 NumPy 数组转换为 PIL 图像
        if isinstance(depth_image, np.ndarray):
            depth_image = Image.fromarray(depth_image)
        return self.dep_transform(depth_image)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/demo0822_steel_all.yaml')
    parser.add_argument('--model', default='save/_demo0822_steel_all/model_epoch_best.pth')
    parser.add_argument('--input_folder', default='load/0822 - all/test/images')
    parser.add_argument('--input_dep', default='load/0822 - all/test/depth') 
    parser.add_argument('--output_folder', default='output/out0822/')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 模型加载
    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda:1')
    model.load_state_dict(sam_checkpoint, strict=True)
    model.eval()
    depth_processor = DepthProcessor(inp_size=1024)


    # 获取文件夹中所有图片的路径
    image_paths = glob.glob(os.path.join(args.input_folder, '*.png'))

    for image_path in image_paths:
        # 读取图片
        inp = cv2.imread(image_path)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(inp, (1024, 1024), interpolation=cv2.INTER_AREA)
        inp = torch.from_numpy(inp).float() / 255.0
        inp = inp.permute(2, 0, 1).unsqueeze(0).cuda()

        #读深度图

        basename = os.path.basename(image_path)
        depth_filename = os.path.join(args.input_dep, basename)
        if os.path.exists(depth_filename):
            dep = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
            dep = depth_processor.process_depth(dep)
            dep = dep.unsqueeze(0).cuda()
        else:
            print(f'Depth map for {basename} not found. Skipping this image.')
            continue

         # 推理
        with torch.no_grad():
            pred_mask = model.infer_depth(inp,dep)
            pred_mask = torch.sigmoid(pred_mask)

        # 转换mask为NumPy数组并保存
        mask_to_display = pred_mask[0, 0].cpu().numpy()
        mask_filename = os.path.join(args.output_folder, os.path.basename(image_path))
        cv2.imwrite(mask_filename, mask_to_display * 255)

        print(f'Mask saved to {mask_filename}')
