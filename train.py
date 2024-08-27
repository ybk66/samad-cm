import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    # 创建数据集对象
    dataset = datasets.make(spec['dataset']) #下
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset}) #上
    if local_rank == 0: #主进程
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[25].items():
            log('  {}: shape={}'.format(k, tuple(v.shape))) 


    # import pdb; pdb.set_trace()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) #在多GPU训练时，每个GPU加载不同的数据子集，避免数据重叠
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler) #8个线程加载数据
    


    return loader # 数据加载器 loader，它可以在训练过程中用于批量读取数据


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')

    return train_loader, val_loader


def eval_psnr(loader, model, eval_type=None):
    model.eval()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    pred_list = []
    gt_list = []
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']
        dep = batch['dep']
        with torch.no_grad():#0606加
            # pred = torch.sigmoid(model.infer(inp))#推理
            pred = torch.sigmoid(model.infer_depth(inp,dep)) #820深度

        batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]

        dist.all_gather(batch_pred, pred)
        pred_list.extend(batch_pred)
        dist.all_gather(batch_gt, batch['gt'])
        gt_list.extend(batch_gt)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    pred_list = torch.cat(pred_list, 1)
    gt_list = torch.cat(gt_list, 1)
    result1, result2, result3, result4 = metric_fn(pred_list, gt_list)

    return result1, result2, result3, result4, metric1, metric2, metric3, metric4


def prepare_training():
    #是否要从之前的训练状态恢复
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])#优化器
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')#最大epoch
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))#记录模型参数数量
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model):
    model.train()#训练模式

    # if local_rank == 0:#是本地主进程，则创建一个进度条
    #     pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    # else:
    #     pbar = None

    loss_list = []
    for batch in train_loader:#遍历训练集，将批次数据转移到指定的设备上
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        dep = batch['dep'] #0820深度
        model.set_input(inp, gt, dep)#设置模型的输入和目标
        model.optimize_parameters()#模型优化参数，即进行一侧前向和后向传播
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)# 优化参数后，收集每个进程的损失
        loss_list.extend(batch_loss)#添加到损失列表中
        # if pbar is not None:
        #     pbar.update(1)

    # if pbar is not None:
    #     pbar.close()

    loss = [i.item() for i in loss_list] 
    return mean(loss) #损失列表的平均值


def main(config_, save_path, args):
    #全局变量初始化
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    #保存配置文件
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    #数据加载
    train_loader, val_loader = make_data_loaders()#数据加载、并行加载模型参数

    #模型参数初始化
    if config.get('data_norm') is None: 
        # 不走这里
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]},
            'dep': {'sub': [0], 'div': [1]}
        }

    #这里就给模型了
    model, optimizer, epoch_start, lr_scheduler = prepare_training()#准备模型、优化器、学习率调度器
    model.optimizer = optimizer
    #设置学习率调度器
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    #模型转到GPU，并设置为分布式数据并行模式
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    #加载SAM的ckpt
    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)#指定是否严格匹配模型和state_dict中的参数键,要插入所以否
    
    #冻结！禁用部分参数的梯度计算
    for name, para in model.named_parameters():#允许查看每个参数的名称，并对它们进行操作
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)#冻结包含image_encoder的，但不冻结包含prompt_generator的部分
        # else:
        #     print(f'{name} 未被冻结')
    

    #如果是主进程，则统计模型参数数量
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())#总参数数量
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)#需要更新的参数数量
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
    #配置训练和验证的参数
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()

    # epoch开始！
    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)#设置当前epoch
        t_epoch_start = timer.t() #开始时间
        train_loss_G = train(train_loader, model)#训练模型，返回损失的平均值
        lr_scheduler.step()#更新学习率
        
        if local_rank == 0:#如果是主进程，则记录训练日志和保存模型
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)#tb记录学习率
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)#tb记录损失
            #####
            writer.flush()#刷新数据写入器，确保日志立即写入
            log(' | '.join(log_info))
            #####

            #保存模型和优化器状态
            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last') #更新最后一个模型

        #如果是验证epoch，则验证模型
        if (epoch_val is not None) and (epoch % epoch_val == 0):
            #验证模型
            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(val_loader, model,
                eval_type=config.get('eval_type'))

            #如果是主进程，则记录验证结果并更新最佳模型
            if local_rank == 0:
                log_info.append('val: {}={:.4f}'.format(metric1, result1))#tb记录验证的性能
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)
                #根据评估类型更新最佳模型
                if config['eval_type'] != 'ber':# 不是阴影检测
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model, save_path, 'best')#
                else:# 是阴影检测，更新第3项，越小越好
                    if result3 < max_val_v:
                        max_val_v = result3
                        print(f"第{epoch}个epoch被视为最优模型，保存！")
                        save(config, model, save_path, 'best')#保存最优的模型

                #计算和记录训练时间、进度等信息
                t = timer.t()#当前时间
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)#当前epoch训练花的时间
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)#已用时间；预计总时间
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))#记录训练时间、进度等信息

                log(', '.join(log_info))#记录日志信息
                writer.flush()#刷新数据写入器，确保日志立即写入


def save(config, model, save_path, name): #模型保存和加载
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            # 保存segformer或setr模型在'evp'模式下的prompt_generator和decode_head
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            # 保存segformer或setr模型在非'evp'模式下的的整个状态字典
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        # 保存整个状态字典
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
        # torch.save(model, os.path.join(save_path, f"model_epoch_{name}.pth"))#保存网络结构
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
