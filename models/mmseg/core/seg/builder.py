from mmcv.utils import Registry, build_from_cfg
# 构建语义分割模型的像素采样器
PIXEL_SAMPLERS = Registry('pixel sampler')


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    return build_from_cfg(cfg, PIXEL_SAMPLERS, default_args)
