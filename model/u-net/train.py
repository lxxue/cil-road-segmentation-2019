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
from torch.nn.modules.batchnorm import BatchNorm2d 

from config import config
from dataloader import get_train_loader
from network import Network_UNet
from datasets import Cil

from utils.init_func import init_weight, group_weight
from engine.engine import Engine


parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, Cil)
    # load model    
    model = Network_UNet(out_planes=config.num_classes, is_training=True)
    # initialize parameters
    init_weight(model.layers, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    base_lr = config.lr
    # group weight initialization on all layers
    params_list = []
    params_list = group_weight(params_list, model.resnet,
                               BatchNorm2d, base_lr*10)
    params_list = group_weight(params_list, model.refine_512,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.refine_256,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.refine_128,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.refine_64,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.up_512,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.up_256,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.up_128,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.up_final,
                               BatchNorm2d, base_lr)

    optimizer = torch.optim.Adam(params_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
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

            lr = optimizer.param_groups[0]['lr'] 

            loss.backward()
            optimizer.step()
            # print information
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % loss

            pbar.set_description(print_str, refresh=False)

        if (epoch%5==0):
            engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            
