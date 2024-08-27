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

def visualize(cfg):
    cfg = utils.load_config(cfg)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/demo0820_steel_light_all.yaml')
    parser.add_argument('--model', default='save/_demo0820_steel_light_all/model_epoch_best.pth')
    parser.add_argument('--input_image', default='load/0820 - all/test/images/962_1.png')  # 修改为单个图像的路径
    parser.add_argument('--input_dep', default='load/0820 - all/test/depth/962_1.png')  
    parser.add_argument('--output_folder', default='output/out0822/')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 模型加载
    model = models.make(config['model']).cuda() 
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=False) 
    
    with open('layer_names.txt', 'w') as f:
        for name, _ in model.named_modules():
            f.write(name + '\n')

    print("Layer names have been saved to 'layer_names.txt'.")
    
    model.eval()

    # 读取单个图像
    inp = cv2.imread(args.input_image)
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(inp, (1024, 1024), interpolation=cv2.INTER_AREA)
    inp = torch.from_numpy(inp).float() / 255.0
    inp = inp.permute(2, 0, 1).unsqueeze(0).cuda() # torch.Size([1, 3, 1024, 1024])

    dep = cv2.imread(args.input_dep, cv2.IMREAD_UNCHANGED)  # 使用 IMREAD_UNCHANGED 以保持深度图像的原始格式
    dep = torch.from_numpy(dep).float() / 255.0  # 将数值范围归一化到 [0, 1]，可根据需要调整
    dep = dep.unsqueeze(0).unsqueeze(0).cuda()  # 调整形状为 [1, 1, 1024, 1024]
    # dep = cv2.resize(dep, (1024, 1024), interpolation=cv2.INTER_AREA)

    # # 将深度图像从 numpy 数组转换为 PyTorch 张量并归一化
    # dep = torch.from_numpy(dep).float() / dep.max()  # 归一化深度值到 [0, 1] 范围
    # dep = dep.unsqueeze(0).unsqueeze(0).cuda()

   
    #手动点point
    # input_point = np.array([[366,366],[453,306],[512,108],[386,401]])
    # input_label = np.array([1,1,1,1])

    # 推理
    with torch.no_grad():
        # pred_mask = model.infer_with_prompt(inp,input_point,input_label)
        pred_mask=model.infer_depth(inp,dep)

        # torch.save(model, os.path.join("存模型", "带连接的model.pth"))#保存网络结构
        # pred_mask = torch.sigmoid(pred_mask)
        pred_mask = torch.clamp(0.2 * pred_mask + 0.5, min=0, max=1) #hard sigmoid 更硬的二值化


    # print("masksshape",pred_mask.shape)

    # 转换mask为NumPy数组并保存
    mask_to_display = pred_mask[0, 0].cpu().numpy() #取第一个维度的第一个元素（索引为 0），然后获取该元素的第一个维度的第一个元素（索引为 0）。这样取得的部分是一个二维数组，通常表示为一个矩阵。
    mask_filename = os.path.join(args.output_folder, os.path.basename(args.input_image))
    cv2.imwrite(mask_filename, mask_to_display * 255) #从0-1映射到0-255

    print(f'Mask saved to {mask_filename}')
    
    # # 保存多个mask
    # for channel in range(3):
    #     mask_to_display=pred_mask[0,channel].cpu().numpy()
    #     mask_filename = os.path.join(args.output_folder, f"{os.path.splitext(os.path.basename(args.input_image))[0]}_channel_{channel}.png")
    #     cv2.imwrite(mask_filename, mask_to_display * 255)
    #     print(f'{channel}通道保存')