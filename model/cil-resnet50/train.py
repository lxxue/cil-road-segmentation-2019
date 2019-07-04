from __future__ import division
import os.path as osp
import sys
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.modules.batchnorm import BatchNorm2d as Bn2D
# from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader import get_train_loader
from network import Network_Res50
from datasets import Cil

from utils.init_func import init_weight, group_weight
from utils.pyt_utils import all_reduce_tensor, load_model
from engine.lr_policy import PolyLR
from engine.engine import Engine
from apex.parallel import DistributedDataParallel, SyncBatchNorm


parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, Cil)
    # use syncBatchNorm if in distributed training
    BatchNorm2d = Bn2D
    if engine.distributed:
        BatchNorm2d = SyncBatchNorm
    # load model 
    model = Network_Res50(out_planes=config.num_classes, is_training=True, 
                    BN2D=BatchNorm2d)
     # initialize parameters
    init_weight(model.layers, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')


    base_lr = config.lr
    if engine.distributed:
         base_lr = config.lr * engine.world_size
    # group weight initialization on all layers
    params_list = []
    params_list = group_weight(params_list, model.context,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.class_refine,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.context_refine,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.arms,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.ffm,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.refines,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.res_top_refine,
                               BatchNorm2d, base_lr)
    # Use adam optimizer for training
    optimizer = torch.optim.Adam(params_list)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.to(device)
    # model fed during parallel training. 
    if engine.distributed:
        model = DistributedDataParallel(model)
     # register state dictations
    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
 # start training process
    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            

            loss = model(imgs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss,
                                                world_size=engine.world_size)
                
            lr = optimizer.param_groups[0]['lr']

            loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % loss

            pbar.set_description(print_str, refresh=False)

        if (epoch%5==0):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)

            
