# -*- encoding: utf-8 -*-
'''
@File    :   pcb.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 23:24   xin      1.0         None
'''

import torch
import torch.nn as nn
import torchvision

from .backbones.resnet import ResNet, Bottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a


# model
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class BottleClassifier(nn.Module):

    def __init__(self, in_dim, out_dim, relu=True, dropout=True, bottle_dim=512):
        super(BottleClassifier, self).__init__()

        bottle = [nn.Linear(in_dim, bottle_dim)]
        bottle += [nn.BatchNorm1d(bottle_dim)]
        if relu:
            bottle += [nn.LeakyReLU(0.1)]
        if dropout:
            bottle += [nn.Dropout(p=0.5)]
        bottle = nn.Sequential(*bottle)
        bottle.apply(weights_init_kaiming)
        self.bottle = bottle

        classifier = [nn.Linear(bottle_dim, out_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.bottle(x)
        x = self.classifier(x)
        return x


class PCB(nn.Module):

    def __init__(self, num_classes, model_path, backbone='resnet50', part_num=6):
        super(PCB, self).__init__()

        # attributes
        self.part_num = part_num
        self.class_num = num_classes

        if backbone == "resnet50":
            self.base = ResNet(last_stride=1)
        elif backbone == "resnet50_ibn_a":
            self.base = resnet50_ibn_a(last_stride=1)
        self.base.load_param(model_path)

        # backbone and optimize its architecture
        # resnet = torchvision.models.resnet50(pretrained=True)
        self.base.layer4[0].downsample[0].stride = (1,1)
        self.base.layer4[0].conv2.stride = (1,1)

        self.base.avgpool_c = nn.AdaptiveAvgPool2d((part_num, 1))
        dropout = nn.Dropout(p=0.5)
        self.base.avgpool_e = nn.AdaptiveAvgPool2d((part_num, 1))

        # cnn feature
        self.resnet_conv = nn.Sequential(
            self.base.conv1, self.base.bn1, self.base.relu, self.base.maxpool,
            self.base.layer1, self.base.layer2, self.base.layer3, self.base.layer4)
        self.pool_c = nn.Sequential(self.base.avgpool_c, dropout)
        self.pool_e = nn.Sequential(self.base.avgpool_e)

        # classifier
        for i in range(part_num):
            name = 'classifier' + str(i)
            setattr(self, name, BottleClassifier(2048, self.class_num, relu=True, dropout=False, bottle_dim=256))

        # embedding
        for i in range(part_num):
            name = 'embedder' + str(i)
            setattr(self, name, nn.Linear(2048, 256))


    def forward(self, x):

        features = self.resnet_conv(x)
        features_c = torch.squeeze(self.pool_c(features))
        features_e = torch.squeeze(self.pool_e(features))

        logits_list = []
        for i in range(self.part_num):
            if self.part_num == 1:
                features_i = features_c
            else:
                features_i = torch.squeeze(features_c[:, :, i])
            classifier_i = getattr(self, 'classifier'+str(i))
            logits_i = classifier_i(features_i)
            logits_list.append(logits_i)

        embeddings_list = []
        for i in range(self.part_num):
            if self.part_num == 1:
                features_i = features_e
            else:
                features_i = torch.squeeze(features_e[:, :, i])
            embedder_i = getattr(self, 'embedder'+str(i))
            embedding_i = embedder_i(features_i)
            embeddings_list.append(embedding_i)
        if self.training:
            return features, features_c, features_e, logits_list, embeddings_list
        else:
            return features_c
