# 批量归一化（Batch Normalization）辅助模块
import torch
import functools
#根据torch版本的不同，它兼容了不同的版本对批量归一化的实现
if torch.__version__.startswith('0'):
    from .sync_bn.inplace_abn.bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    BatchNorm2d_class = InPlaceABNSync
    relu_inplace = False
else:
    BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
    relu_inplace = True

import torch
BatchNorm2d = torch.nn.BatchNorm2d
BatchNorm2d_class = BatchNorm2d
relu_inplace = False