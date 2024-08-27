import torch
import torch.nn as nn
import cv2
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def load_and_preprocess_images(rgb_path, depth_path):
    # 读取RGB图像和Depth图像
    rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    # 将图像转换为torch tensors
    rgb_tensor = torch.from_numpy(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float()
    depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).float()

    return rgb_tensor, depth_tensor

def main(rgb_path, depth_path):
    # 加载和预处理图像
    rgb_tensor, depth_tensor = load_and_preprocess_images(rgb_path, depth_path)

    # 初始化特征提取器
    rgb_feature_extractor = FeatureExtractor(in_channels=3)
    depth_feature_extractor = FeatureExtractor(in_channels=1)

    # 通过特征提取器提取特征 f1 和 f2
    f1 = rgb_feature_extractor(rgb_tensor)  # 处理 RGB 图像
    f2 = depth_feature_extractor(depth_tensor)  # 处理 Depth 图像

    # 进行 element-wise 乘法得到 f3
    f3 = f1 * f2

    # 将 f1, f2, f3 进行 concat
    fused_features = torch.cat((f1, f2, f3), dim=1)  # 在通道维度上拼接

    print(f"f1 shape: {f1.shape}")
    print(f"f2 shape: {f2.shape}")
    print(f"f3 shape: {f3.shape}")
    print(f"Fused feature shape: {fused_features.shape}")

    return fused_features

# 使用本地图片路径调用主函数
fused_features = main('load/0818 - visonly/train/images/8_2.png', '8_2.png')


