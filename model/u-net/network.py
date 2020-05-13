import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from config import config
from torch.nn.modules.batchnorm import BatchNorm2d as BN2D
from base_model import resnet18,resnet34,resnet50
from seg_opr.seg_oprs import AttentionRefinement, FeatureFusion

class Network_UNet(nn.Module):
    def __init__(self, out_planes, is_training):
        super(Network_UNet, self).__init__()
        self.layers = []
        self.is_training = is_training
        
        self.layers = []
        
        conv_channel = 128
        # base model of resnet 18 from resnet.py
        self.resnet = resnet18(pretrained_model=None, norm_layer=BN2D, bn_eps=config.bn_eps, bn_momentum=config.bn_momentum, deep_stem=False, stem_width=64)
        # tail refinement on FCN style contraction path
        self.refine_512 = nn.Sequential(
                        ConvBnRelu(512, 1024, 7, 1, 3, has_bn=False, has_relu=True, has_bias=False, norm_layer=BN2D),
                        nn.Dropout2d(),
                        ConvBnRelu(1024, out_planes, 1, 1, 0, has_bn=False, has_relu=True, has_bias=False, norm_layer=BN2D),
                        nn.Dropout2d(),
                        nn.AdaptiveAvgPool2d(1)
        )
        
        # upscale using Transpose convolution
        self.up_512 = nn.ConvTranspose2d(out_planes, out_planes, kernel_size=4,stride=2,padding=2, output_padding=1)
        # Refinement on intermediate layers in FCN style structure
        self.refine_256 = ConvBnRelu(256, out_planes, 1, 1, 0, has_bn=False, has_relu=False, has_bias=False, norm_layer=BN2D)
        self.refine_128 = ConvBnRelu(128, out_planes, 1, 1, 0, has_bn=False, has_relu=False, has_bias=False, norm_layer=BN2D)
        self.refine_64 = ConvBnRelu(64, out_planes, 1, 1, 0, has_bn=False, has_relu=False, has_bias=False, norm_layer=BN2D) 
        
        # upscale using Transpose convolution
        self.up_256 = nn.ConvTranspose2d(out_planes, out_planes, kernel_size=4,stride=2,padding=1)
        self.up_128 = nn.ConvTranspose2d(out_planes, out_planes, kernel_size=4,stride=2,padding=1)

        
        self.up_final = nn.ConvTranspose2d(out_planes, out_planes, kernel_size=8,stride=4,padding=2)           


        self.layers.append(self.resnet)
        self.layers.append(self.refine_512)
        self.layers.append(self.refine_256)
        self.layers.append(self.refine_128)
        self.layers.append(self.refine_64)
        self.layers.append(self.up_512)
        self.layers.append(self.up_256)
        self.layers.append(self.up_128)
        self.layers.append(self.up_final)

        self.loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)	
        
    def forward(self, x, gt=None):
        print(x.shape)
        resnet_out = self.resnet(x)
        resnet_out.reverse()
        
        refine_512 = self.refine_512(resnet_out[0])
        refine_256 = self.refine_256(resnet_out[1])
        refine_128 = self.refine_128(resnet_out[2])
        refine_64 = self.refine_64(resnet_out[3])

        refine_512 = F.interpolate(refine_512, size=(resnet_out[0].size()[2:]),
                                mode='bilinear', align_corners=True)
        up_512 = self.up_512(refine_512)

        fuse_8 = up_512 + refine_256
        up_256 = self.up_256(fuse_8)

        fuse_4 = up_256 + refine_128
        up_128 = self.up_128(fuse_4)
        fuse_2 = up_128 + refine_64
        result = self.up_final(fuse_2)

        
        if self.is_training:
            loss = self.loss(result,gt)
            return loss
        
        return F.log_softmax(result,dim=1)
    
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5, 
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)
            
    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
            
        return x






