from PIL import Image
import torch
from torchvision import transforms

# 加载图片
image_path = 'load/0820 - all/train/masks/1_1.png'  # 替换为你的图片路径
image = Image.open(image_path)

# 打印原始图片的形状
print(f"Original image size: {image.size}")  # PIL的图片大小是 (宽度, 高度)

# 如果你想将图片转换为Tensor
transform = transforms.ToTensor()
image_tensor = transform(image)

# 打印Tensor的形状
print(f"Image tensor shape: {image_tensor.shape}")  # 打印形状 (C, H, W)
