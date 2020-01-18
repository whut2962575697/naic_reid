# -*- encoding: utf-8 -*-
'''
@File    :   mgn.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 18:33   xin      1.0         None
'''

import copy

import torch
from torch import nn
import torch.nn.functional as F

from .backbones.resnet import ResNet, Bottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.resnext_ibn_a import resnext101_ibn_a
from .layers.pooling import GeM,GlobalConcatPool2d,GlobalAttnPool2d,GlobalAvgAttnPool2d,GlobalMaxAttnPool2d,GlobalConcatAttnPool2d,AdaptiveGeM2d


class MGN(nn.Module):
    def __init__(self, num_classes, model_path, last_stride=1,backbone='resnet50', pool_type='max',part_pool_type='max',use_center=False,num_share_layer3=1,use_bnbias=True):
        super(MGN, self).__init__()

        self.use_center = use_center
        if backbone == "resnet50":
            self.base = ResNet(last_stride=last_stride)
        elif backbone == "resnet50_ibn_a":
            self.base = resnet50_ibn_a(last_stride=last_stride)
        else:
            self.base = eval(backbone)(last_stride=last_stride)
        self.base.load_param(model_path)

  
        self.backbone = nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3[:num_share_layer3]
        )

        res_conv4 = nn.Sequential(*self.base.layer3[num_share_layer3:])


        res_g_conv5 = self.base.layer4

        res_p_conv5 = copy.deepcopy(self.base.layer4)
        for n, m in res_p_conv5.named_modules():
            if 'conv2' in n:
                m.stride =  (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
       
        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))


        if pool_type == "avg":
            pool2d_p1 = nn.AdaptiveAvgPool2d(1)
            pool2d_p2 = nn.AdaptiveAvgPool2d(1)
            pool2d_p3 = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'max':
            pool2d_p1 = nn.AdaptiveMaxPool2d(1)
            pool2d_p2 = nn.AdaptiveMaxPool2d(1)
            pool2d_p3 = nn.AdaptiveMaxPool2d(1)
        elif "gem" in pool_type:
            if pool_type !='gem':
                p = pool_type.split('_')[-1]
                p = float(p)
                pool2d_p1 = GeM(p=p, eps=1e-6, freeze_p=True)
                pool2d_p2 = GeM(p=p, eps=1e-6, freeze_p=True)
                pool2d_p3 = GeM(p=p, eps=1e-6, freeze_p=True)
            else:
                pool2d_p1 = GeM(eps=1e-6, freeze_p=False)
                pool2d_p2 = GeM(eps=1e-6, freeze_p=False)
                pool2d_p3 = GeM(eps=1e-6, freeze_p=False)

        if part_pool_type == "avg":
            pool2d_zp2 = nn.AdaptiveAvgPool2d((2,1))
            pool2d_zp3 = nn.AdaptiveAvgPool2d((3,1))

        elif part_pool_type == 'max':
            pool2d_zp2 = nn.AdaptiveMaxPool2d((2,1))
            pool2d_zp3 = nn.AdaptiveMaxPool2d((3,1))

        elif "gem" in part_pool_type:
            if part_pool_type !='gem':
                p = part_pool_type.split('_')[-1]
                p = float(p)
                pool2d_zp2 = AdaptiveGeM2d(output_size=(2,1),p=p, eps=1e-6, freeze_p=True)
                pool2d_zp3 = AdaptiveGeM2d(output_size=(3,1),p=p, eps=1e-6, freeze_p=True)

            else:
                pool2d_zp2 = AdaptiveGeM2d(output_size=(2,1),eps=1e-6, freeze_p=False)
                pool2d_zp3 = AdaptiveGeM2d(output_size=(3,1),eps=1e-6, freeze_p=False)


        self.maxpool_zg_p1 = pool2d_p1
        self.maxpool_zg_p2 = pool2d_p2
        self.maxpool_zg_p3 = pool2d_p3
        self.maxpool_zp2 = pool2d_zp2
        self.maxpool_zp3 = pool2d_zp3
        
        # if pool_type == 'max':
        #     pool2d = nn.AdaptiveMaxPool2d
        # elif pool_type == 'avg':
        #     pool2d = nn.AdaptiveAvgPool2d
        # else:
        #     raise Exception()

        # self.maxpool_zg_p1 = pool2d(output_size=(1, 1))
        # self.maxpool_zg_p2 = pool2d(output_size=(1, 1))
        # self.maxpool_zg_p3 = pool2d(output_size=(1, 1))
        # self.maxpool_zp2 = pool2d(output_size=(2, 1))
        # self.maxpool_zp3 = pool2d(output_size=(3, 1))

        if use_bnbias:
            reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256),nn.PReLU(256, 0.25))
        else:
            reduction = nn.Sequential(nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(256))
            reduction[1].bias.requires_grad_(False)

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(256, num_classes)
        self.fc_id_2048_1 = nn.Linear(256, num_classes)
        self.fc_id_2048_2 = nn.Linear(256, num_classes)

        self.fc_id_256_1_0 = nn.Linear(256, num_classes)
        self.fc_id_256_1_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_0 = nn.Linear(256, num_classes)
        self.fc_id_256_2_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_2 = nn.Linear(256, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
        if self.training:
            if self.use_center:
                return fg_p1, fg_p2, fg_p3, \
                        l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3, \
                        f0_p2, f1_p2, f0_p3, f1_p3, f2_p3
            else:
                return fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3
        else:
            return predict