# encoding: utf-8
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from base_model import resnet50
from seg_opr.seg_oprs import ConvBnRelu


class PSPNet(nn.Module):
    def __init__(self, out_planes, criterion=None, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(PSPNet, self).__init__()
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                 bn_eps=config.bn_eps,
                                 bn_momentum=config.bn_momentum,
                                 deep_stem=True, stem_width=64)
        self.backbone.layer3.apply(partial(self._nostride_dilate, dilate=2))
        self.backbone.layer4.apply(partial(self._nostride_dilate, dilate=4))

        self.business_layer = []
        self.psp_layer = PyramidPooling('psp', out_planes, 2048,
                                        norm_layer=norm_layer)
        self.aux_layer = nn.Sequential(
            ConvBnRelu(1024, 1024, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(1024, out_planes, kernel_size=1)
        )
        self.business_layer.append(self.psp_layer)
        self.business_layer.append(self.aux_layer)

        self.criterion = criterion

        ngf = 64
        ngf2 = ngf//2
        use_selu = True
        in_nc = 3
        out_nc = 2
        #------------road edge detection------------#
        self.edge_conv1 = nn.Sequential(*self._conv_block(in_nc+out_nc, ngf2, norm_layer, use_selu, num_block=2))
        self.side_edge_conv1 = nn.Conv2d(ngf2, out_nc, kernel_size=1, stride=1, bias=False) 

        self.edge_conv2 = nn.Sequential(*self._conv_block(ngf2, ngf2*2, norm_layer, use_selu, num_block=2))
        self.side_edge_conv2 = nn.Conv2d(ngf2*2, out_nc, kernel_size=1, stride=1, bias=False)

        self.edge_conv3 = nn.Sequential(*self._conv_block(ngf2*2, ngf2*4, norm_layer, use_selu, num_block=2))
        self.side_edge_conv3 = nn.Conv2d(ngf2*4, out_nc, kernel_size=1, stride=1, bias=False)

        self.edge_conv4 = nn.Sequential(*self._conv_block(ngf2*4, ngf2*8, norm_layer, use_selu, num_block=2))
        self.side_edge_conv4 = nn.Conv2d(ngf2*8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_edge_conv = nn.Conv2d(out_nc*4, out_nc, kernel_size=1, stride=1, bias=False)

        #------------road centerline extraction------------#
        self.centerline_conv1 = nn.Sequential(*self._conv_block(in_nc+out_nc, ngf2, norm_layer, use_selu, num_block=2))
        self.side_centerline_conv1 = nn.Conv2d(ngf2, out_nc, kernel_size=1, stride=1, bias=False) 

        self.centerline_conv2 = nn.Sequential(*self._conv_block(ngf2, ngf2*2, norm_layer, use_selu, num_block=2))
        self.side_centerline_conv2 = nn.Conv2d(ngf2*2, out_nc, kernel_size=1, stride=1, bias=False)

        self.centerline_conv3 = nn.Sequential(*self._conv_block(ngf2*2, ngf2*4, norm_layer, use_selu, num_block=2))
        self.side_centerline_conv3 = nn.Conv2d(ngf2*4, out_nc, kernel_size=1, stride=1, bias=False)

        self.centerline_conv4 = nn.Sequential(*self._conv_block(ngf2*4, ngf2*8, norm_layer, use_selu, num_block=2))
        self.side_centerline_conv4 = nn.Conv2d(ngf2*8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_centerline_conv = nn.Conv2d(out_nc*4, out_nc, kernel_size=1, stride=1, bias=False)

        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.business_layer.append(self.edge_conv1)
        self.business_layer.append(self.edge_conv2)
        self.business_layer.append(self.edge_conv3)
        self.business_layer.append(self.edge_conv4)
        self.business_layer.append(self.side_edge_conv1)
        self.business_layer.append(self.side_edge_conv2)
        self.business_layer.append(self.side_edge_conv3)
        self.business_layer.append(self.side_edge_conv4)
        self.business_layer.append(self.fuse_edge_conv)


        self.business_layer.append(self.centerline_conv1)
        self.business_layer.append(self.centerline_conv2)
        self.business_layer.append(self.centerline_conv3)
        self.business_layer.append(self.centerline_conv4)
        self.business_layer.append(self.side_centerline_conv1)
        self.business_layer.append(self.side_centerline_conv2)
        self.business_layer.append(self.side_centerline_conv3)
        self.business_layer.append(self.side_centerline_conv4)
        self.business_layer.append(self.fuse_centerline_conv)


    def _edge_forward(self, x):
        """
        predict road edge
        :param: x, [image tensor, predicted segmentation tensor], [N, C+1, H, W]
        """
        h, w = x.size()[2:]
        # main stream features
        conv1 = self.edge_conv1(x)
        conv2 = self.edge_conv2(self.maxpool(conv1))
        conv3 = self.edge_conv3(self.maxpool(conv2))
        conv4 = self.edge_conv4(self.maxpool(conv3))
        # side output features
        side_output1 = self.side_edge_conv1(conv1)
        side_output2 = self.side_edge_conv2(conv2)
        side_output3 = self.side_edge_conv3(conv3)
        side_output4 = self.side_edge_conv4(conv4)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear', align_corners=True) #self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear', align_corners=True) #self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear', align_corners=True) #self.up8(side_output4)
        fused = self.fuse_edge_conv(torch.cat([
            side_output1, 
            side_output2, 
            side_output3,
            side_output4], dim=1))        
        return [side_output1, side_output2, side_output3, side_output4, fused]

    def _centerline_forward(self, x):
        """
        predict road edge
        :param: x, [image tensor, predicted segmentation tensor], [N, C+1, H, W]
        """
        h,w = x.size()[2:]
        # main stream features
        conv1 = self.centerline_conv1(x)
        conv2 = self.centerline_conv2(self.maxpool(conv1))
        conv3 = self.centerline_conv3(self.maxpool(conv2))
        conv4 = self.centerline_conv4(self.maxpool(conv3))
        # side output features
        side_output1 = self.side_centerline_conv1(conv1)
        side_output2 = self.side_centerline_conv2(conv2)
        side_output3 = self.side_centerline_conv3(conv3)
        side_output4 = self.side_centerline_conv4(conv4)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear', align_corners=True) #self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear', align_corners=True) #self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear', align_corners=True) #self.up8(side_output4)
        fused = self.fuse_centerline_conv(torch.cat([
            side_output1, 
            side_output2, 
            side_output3,
            side_output4], dim=1))
        return [side_output1, side_output2, side_output3, side_output4, fused]



    def forward(self, data, label=None, edge_label=None, centerline_label=None):
        blocks = self.backbone(data)

        psp_fm = self.psp_layer(blocks[-1])
        aux_fm = self.aux_layer(blocks[-2])

        psp_fm = F.interpolate(psp_fm, scale_factor=8, mode='bilinear',
                               align_corners=True)
        aux_fm = F.interpolate(aux_fm, scale_factor=8, mode='bilinear',
                               align_corners=True)

        x = torch.cat([data, psp_fm], dim=1)
        edge_fm = self._edge_forward(x)[-1]
        centerline_fm = self._centerline_forward(x)[-1]

        psp_fm = F.log_softmax(psp_fm, dim=1)
        aux_fm = F.log_softmax(aux_fm, dim=1)
        edge_fm = F.log_softmax(edge_fm)
        centerline_fm = F.log_softmax(centerline_fm)

        if label is not None:
            # print(label.min(), label.max())
            loss = self.criterion(psp_fm, label)
            aux_loss = self.criterion(aux_fm, label)
            edge_loss = self.criterion(edge_fm, edge_label) 
            centerline_loss = self.criterion(centerline_fm, centerline_label)
            print("loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(loss.item(), aux_loss.item(), edge_loss.item(), centerline_loss.item()))
            loss = loss + 0.4 * aux_loss + 0.2 * edge_loss + 0.2 * centerline_loss
            return loss

        return psp_fm

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def _conv_block(self, in_nc, out_nc, norm_layer, use_selu, num_block=2, kernel_size=3, 
        stride=1, padding=1, bias=True):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride, 
                               padding=padding, bias=bias)]
            if use_selu:
                conv += [nn.SELU(True)]
            else:
                conv += [norm_layer(out_nc), nn.ReLU(True)]
        return conv


class PyramidPooling(nn.Module):
    def __init__(self, name, out_planes, fc_dim=4096, pool_scales=[1, 2, 3, 6],
                 norm_layer=nn.BatchNorm2d):
        super(PyramidPooling, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(OrderedDict([
                ('{}/pool_1'.format(name), nn.AdaptiveAvgPool2d(scale)),
                ('{}/cbr'.format(name),
                 ConvBnRelu(fc_dim, 512, 1, 1, 0, has_bn=True,
                            has_relu=True, has_bias=False,
                            norm_layer=norm_layer))
            ])))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv6 = nn.Sequential(
            ConvBnRelu(fc_dim + len(pool_scales) * 512, 512, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(512, out_planes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        for pooling in self.ppm:
            ppm_out.append(
                F.interpolate(pooling(x), size=(input_size[2], input_size[3]),
                              mode='bilinear', align_corners=True))
        ppm_out = torch.cat(ppm_out, 1)

        ppm_out = self.conv6(ppm_out)
        return ppm_out


if __name__ == "__main__":
    model = PSPNet(150, None)
    print(model)
