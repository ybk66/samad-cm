import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import matplotlib.pyplot as plt


from models import register

#注意：没有加入提示编码器
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer, PromptEncoder

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple

from .mmseg.models.utils.tranforms import ResizeLongestSide


def init_weights(layer):#初始化权重，输入：要初始化的层
    if type(layer) == nn.Conv2d:#卷积层
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:#线性层
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:#批归一化层
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    平衡的二元交叉熵损失函数
    用于二分类问题中不平衡数据集的情况
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):#iou损失函数
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    利用随机的空间频率的位置编码
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(#用一个缓存区存储随机生成的高斯矩阵，用于位置编码
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),#缩放生成的高斯矩阵
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        #对坐标进行位置编码
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        # 首先将坐标缩放到 [-1, 1] 的范围内，然后与高斯矩阵相乘，最后乘以2π
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)#坐标的正弦和余弦值，并将它们在最后一个维度上拼接起来。


    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        # "为指定大小的网格生成位置编码
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W #为模型提供关于图像中像素位置的信息


@register('sam')#装饰器：将特定的模型类注册到models字典中，并通过与'sam'相对应的键进行关联。
#允许在不修改原始代码的情况下添加功能或逻辑
#接下来会对SAM模型做修改
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #从encoder_mode中提取维度
        self.embed_dim = encoder_mode['embed_dim']
        #初始化图像编码器
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        #从encoder_mode中提取维度，初始化提示嵌入维度
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        #初始化掩码解码器
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        image_embedding_size=inp_size//encoder_mode['patch_size']
        image_size=inp_size

        ###6.13想加提示编码器###
        # self.prompt_encoder=PromptEncoder(
        #     embed_dim=self.prompt_embed_dim,
        #     image_embedding_size=(image_embedding_size, image_embedding_size),
        #     input_image_size=(image_size, image_size),
        #     mask_in_chans=16,
        # )


        #如果编码器模式“evp”（不），则冻结不包含prompt、mask_decoder、prompt_encoder的参数
        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
                    p.requires_grad = False


        #初始化损失函数
        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()
        # 初始化位置嵌入层
        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        #初始化图像嵌入尺寸
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        #初始化一个不带掩码的嵌入层
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])
        #613新加
        self.original_size = [inp_size,inp_size]
        self.transform = ResizeLongestSide(inp_size)


    def set_input(self, input, gt_mask, dep):#将图片和对应的GT放到指定设备
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)
        self.dep = dep.to(self.device) #820深度

    def get_dense_pe(self) -> torch.Tensor:
        """
        返回用于编码点提示的位置编码，应用于密集点集，这些点的形状与图像编码的形状相同。

        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def save_featmap(mask):
        mask= mask.squeeze().cpu().numpy()
        plt.imshow(mask, cmap='gray')
        plt.axis('off')  # 不显示坐标轴
        # 保存图像为文件
        plt.savefig('my_image.png', bbox_inches='tight', pad_inches=0.0)
        plt.close()
        print("图像已保存为 my_image.png")
    def forward(self):
        bs = 1

        # Embed prompts
        # 初始化稀疏提示嵌入和密集提示嵌入
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        #通过图像编码器提取图像特征
        self.features = self.image_encoder(self.input, self.dep) #820深度

        # Predict masks
        #通过掩码解码器预测掩码和IOU
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        # 对低分辨率掩码进行上采样，以匹配输入图像的分辨率
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks

    def infer(self, input):
        #进行推理生成掩码
        bs = 1
        self.features = self.image_encoder(input)

        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        # Predict masks 预测掩码
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution 上采样到原图尺寸
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks
    def infer_depth(self,input,dep):
                #进行推理生成掩码
        bs = 1
        self.features = self.image_encoder(input,dep)

        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        # Predict masks 预测掩码
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution 上采样到原图尺寸
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks
    
    def postprocess_masks(#掩码后处理
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.
        移除padding并将掩码上采样到原始图像大小

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        
        # save_featmap(masks)
        # torch.save(masks, '刚进后处理时的特征图.pt')
        #torch.Size([1, 1, 256, 256])
        masks = F.interpolate(#双线性插值，上采样到统一尺寸1024*1024
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),#目标尺寸
            mode="bilinear",
            align_corners=False,
        )
        # torch.Size([1, 1, 1024, 1024]) 
        masks = masks[..., : input_size, : input_size]# 移除pad，保留了 masks 张量的  最后两个维度  的  input_size 个元素。
        #torch.Size([1, 1, 1024, 1024])
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)#上采样到最原始的图像的大小，由于设置成1024了，所以看似没变化
        #torch.Size([1, 1, 1024, 1024])
        return masks

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # GAN和L1损失
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()# 前向传播
        self.optimizer.zero_grad()  # set G's gradients to zero 梯度置零
        self.backward_G()  # calculate graidents for G 计算梯度
        self.optimizer.step()  # udpate G's weights 更新优化器的参数

    def set_requires_grad(self, nets, requires_grad=False):
        """
        为给定的网络列表设置参数的求导需求，默认为不求导。
        这个函数主要用于避免不必要的计算。
        """
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def infer_with_prompt(self,input,point_coords,point_labels):
        # def save_featmap(mask):
        #     mask= mask.squeeze().cpu().numpy()
        #     plt.imshow(mask, cmap='gray')
        #     plt.axis('off')  # 不显示坐标轴
        #     # 保存图像为文件
        #     plt.savefig('my_image.png', bbox_inches='tight', pad_inches=0.0)
        #     plt.close()
        #     print("图像已保存为 my_image.png")

        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            points = (coords_torch, labels_torch)
        
        #进行推理生成掩码
        bs = 1
        self.features = self.image_encoder(input)

        ###6.13搞一个稀疏提示
        sparse_embeddings,dense_embeddings=self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        # sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        # dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            # bs, -1, self.image_embedding_size, self.image_embedding_size
        # ) #这个东西起到了关键的作用，即能够生成比较好的mask而不是点阵

        # Predict masks 预测掩码
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution 上采样到原图尺寸
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks
        
    
    def infer_with_multimasks(self, input):
        #进行推理生成掩码
        bs = 1
        self.features = self.image_encoder(input)

        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        # Predict masks 预测掩码
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        # Upscale the masks to the original image resolution 上采样到原图尺寸
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks
