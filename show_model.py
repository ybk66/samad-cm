import torch
import torch.nn as nn
import torchvision.models as models

# 定义你的模型（需要与保存的.pt文件中的模型架构匹配）
model = models.resnet18()  # 示例：使用ResNet18模型

# 从本地加载预训练的.pt文件
model_weights_path = '存模型/带连接的model.pth'  # 替换为实际的.pt文件路径
model.load_state_dict(torch.load(model_weights_path))

# 将模型设为评估模式（一般在推理时使用）
model.eval()

# 保存所有层的名字到一个.txt文件
with open('layer_names.txt', 'w') as f:
    for name, _ in model.named_modules():
        f.write(name + '\n')

print("Layer names have been saved to 'layer_names.txt'.")
