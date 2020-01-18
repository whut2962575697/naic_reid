# -*- encoding: utf-8 -*-
'''
@File    :   small_mhn_pcb.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/7 10:33   xin      1.0         None
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

from .modules.HighDivModule import HighDivModule
from .backbones.resnet import ResNet, Bottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class MHN_smallPCB(nn.Module):
    def __init__(self, class_num, model_path, backbone='resnet50', parts=4, part=6):
        super(MHN_smallPCB, self).__init__()

        self.part = part  # We cut the pool5 to 6 part
        if backbone == "resnet50":
            self.base = ResNet(last_stride=1)
        elif backbone == "resnet50_ibn_a":
            self.base = resnet50_ibn_a(last_stride=1)
        self.base.load_param(model_path)
        # model_ft = models.resnet50(pretrained=True)
        self.model = self.base
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        self.parts = parts

        for i in range(self.parts):
            name = 'HIGH' + str(i)
            setattr(self, name, HighDivModule(512, i + 1))

        # define 6 fc
        for i in range(self.part):
            name = 'fc' + str(i)
            setattr(self, name, nn.Sequential(nn.Linear(2048, 256), nn.BatchNorm1d(256)))
        for i in range(self.part):
            name = 'fc' + str(i)
            layer = getattr(self, name)
            layer.apply(weights_init_kaiming)

        # define 6*parts classifiers
        for i in range(self.part):
            for j in range(self.parts):
                name = 'classifier' + str(i) + '_' + str(j)
                setattr(self, name, nn.Sequential(nn.Linear(256, class_num)))
        for i in range(self.part):
            for j in range(self.parts):
                name = 'classifier' + str(i) + '_' + str(j)
                layer = getattr(self, name)
                layer.apply(weights_init_classifier)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)

        x_ = []
        for i in range(self.parts):
            name = 'HIGH' + str(i)
            layer = getattr(self, name)
            x_.append(layer(x))

        x = torch.cat(x_, 0)

        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        fea = []
        # get six part feature parts*batchsize,256,part
        for i in range(self.part):
            name = 'fc' + str(i)
            c = getattr(self, name)
            fea.append(F.normalize(c(torch.squeeze(x[:, :, i]))) * 20)  # normalize features and scale to 20

        # get part*parts predict
        y = []
        num = int(fea[0].size(0) / self.parts)
        for i in range(self.part):
            for j in range(self.parts):
                name = 'classifier' + str(i) + '_' + str(j)
                c = getattr(self, name)
                y.append(c(fea[i][j * num:(j + 1) * num, :]))
        return y, fea