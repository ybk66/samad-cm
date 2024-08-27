import cv2
import torch

# 从本地读取 RGB 图像和 Depth 图像
rgb_image_path = 'load/0818 - visonly/train/images/8_2.png'  # 替换为你的RGB图像路径
depth_image_path = '8_2.png'  # 替换为你的Depth图像路径

# 读取RGB图像（默认读取为BGR格式，需要转换为RGB）
rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

# 读取Depth图像
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

# 将图像转换为torch tensors并进行通道检查
rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)
depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).float()  # (H, W) -> (1, H, W)

# 检查通道是否合规
if rgb_tensor.shape[0] != 3:
    raise ValueError(f"RGB图像应具有3个通道，但实际有 {rgb_tensor.shape[0]} 个通道")
if depth_tensor.shape[0] != 1:
    raise ValueError(f"Depth图像应具有1个通道，但实际有 {depth_tensor.shape[0]} 个通道")

# 确保深度图像的尺寸与RGB图像一致
if rgb_tensor.shape[1:] != depth_tensor.shape[1:]:
    raise ValueError("RGB图像和Depth图像的尺寸不匹配")

# 将 RGB 图像和 Depth 图像在通道维度上连接起来
fused_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0)  # 结果是4通道的图像 (4, H, W)

print(fused_tensor.shape)  # 输出结果应为 (4, H, W)

# 如果需要，可以将结果保存为图像
fused_image = fused_tensor.permute(1, 2, 0).numpy().astype('uint8')  # (C, H, W) -> (H, W, C)
cv2.imwrite('fused_image.png', fused_image)  # 保存为png格式
