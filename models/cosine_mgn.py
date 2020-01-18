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
from .layers.cosine_loss import AdaCos,ArcFace,SphereFace,CosFace,ArcCos


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        # nn.init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal(m.weight.data, std=0.001)
        # nn.init.constant(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--bn2d--|--relu--|--conv2d--|--pooling--|--bnneck(triplet-bn-cls)--|
# triplet and cls 
class ClassBlock2D(nn.Module):
    def __init__(self, input_dim, num_feat, num_classes,kernel_size , stride, dropout=False, relu=False,pool_type='avg',use_bnbias=False,cosine_loss_type='',scale=30.0,margin=0.35,reduce_bnfirst=True):
        super(ClassBlock2D, self).__init__()
        self.num_classes = num_classes
        self.cosine_loss_type = cosine_loss_type

        # reduction
        reduction = []
        if reduce_bnfirst:
            reduction += [nn.BatchNorm2d(input_dim)]
            reduction += [nn.ReLU(inplace=True)]
            reduction += [nn.Conv2d(input_dim, num_feat, kernel_size=kernel_size, stride=stride, bias=False)]
        else:
            reduction += [nn.Conv2d(input_dim, num_feat, kernel_size=kernel_size, stride=stride, bias=False)]
            reduction += [nn.BatchNorm2d(num_feat)]
            reduction += [nn.PReLU(num_feat, 0.25)]

        reduction = nn.Sequential(*reduction)
        reduction.apply(weights_init_kaiming)
        self.reduction = reduction
        # pool
        if pool_type == "avg":
            self.pool2d = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'max':
            self.pool2d = nn.AdaptiveMaxPool2d(1)
        elif "gem" in pool_type:
            if pool_type !='gem':
                p = pool_type.split('_')[-1]
                p = float(p)
                self.pool2d = GeM(p=p, eps=1e-6, freeze_p=True)
            else:
                self.pool2d = GeM(eps=1e-6, freeze_p=False)
        elif 'Att' in pool_type:
            self.pool2d = eval(pool_type)(in_features = num_feat)
            num_feat = self.pool2d.out_features(num_feat)
        else:
            self.pool2d = eval(pool_type)()
            num_feat = self.pool2d.out_features(num_feat)
        # bottle
        self.bottleneck = nn.BatchNorm1d(num_feat)
        if use_bnbias == False:
            print('==>remove bnneck bias')
            self.bottleneck.bias.requires_grad_(False)  # no shift
        else:
            print('==>using bnneck bias')
        self.bottleneck.apply(weights_init_kaiming)
        # classify
        if cosine_loss_type=='':
            self.classifier = nn.Linear(num_feat, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        else:
            if cosine_loss_type == 'AdaCos':
                self.classifier = eval(cosine_loss_type)(num_feat, self.num_classes,margin)
            else:
                self.classifier = eval(cosine_loss_type)(num_feat, self.num_classes,scale,margin)

    def forward(self, x,label=None):
        x = self.reduction(x)
        x = self.pool2d(x)
        pool_feat = torch.squeeze(x)
        feat = self.bottleneck(pool_feat)
        if self.training:
            if label is not None:
                cls_score = self.classifier(feat,label)
            else:
                cls_score = self.classifier(feat)
        else:
            cls_score = None
        return pool_feat, feat, cls_score

# cls only
class ClassBlock1D(nn.Module):
    def __init__(self, input_dim,num_feat, num_classes, dropout=False, relu=True,use_bnbias=False,cosine_loss_type='',scale=30.0,margin=0.35):
        super(ClassBlock1D, self).__init__()
        self.num_classes = num_classes
        self.cosine_loss_type = cosine_loss_type

        # # reduction tpl loss is bigger
        # reduction = []
        # reduction += [nn.BatchNorm1d(input_dim)]
        # if relu:
        #     reduction += [nn.ReLU(inplace=True)]
        # reduction += [nn.Linear(input_dim, num_feat, bias=False)]

             # reduction
        reduction = []
        reduction += [nn.Linear(input_dim, num_feat, bias=False)]
        reduction += [nn.BatchNorm1d(num_feat)]
        if relu:
            reduction += [nn.PReLU(num_feat, 0.25)]
            # reduction += [nn.ReLU(inplace=True)]

        if dropout:
            reduction += [nn.Dropout(p=0.5)]

        reduction = nn.Sequential(*reduction)
        reduction.apply(weights_init_kaiming)
        self.reduction = reduction
        
        # bottle
        self.bottleneck = nn.BatchNorm1d(num_feat)
        if use_bnbias == False:
            print('==>remove bnneck bias')
            self.bottleneck.bias.requires_grad_(False)  # no shift
        else:
            print('==>using bnneck bias')

        self.bottleneck.apply(weights_init_kaiming)

        # classify
        if cosine_loss_type=='':
            self.classifier = nn.Linear(num_feat, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        else:
            if cosine_loss_type == 'AdaCos':
                self.classifier = eval(cosine_loss_type)(num_feat, self.num_classes,margin)
            else:
                self.classifier = eval(cosine_loss_type)(num_feat, self.num_classes,scale,margin)

    def forward(self, x,label):
        reduce_feat = self.reduction(x)
        feat = self.bottleneck(reduce_feat)
        if self.training:
            if label is not None:
                cls_score = self.classifier(feat,label)
            else:
                cls_score = self.classifier(feat)
        else:
            cls_score = None
        return reduce_feat, feat, cls_score


class CosineMGN(nn.Module):
    def __init__(self, num_classes, model_path, last_stride=1,backbone='resnet50', \
                        num_share_layer3=1,
                        pool_type='max',part_pool_type='max',\
                        use_center=False,guse_bnbias=True,puse_bnbias=True,\
                        g_cosine_loss_type='',p_cosine_loss_type='',scale=30.0,margin=0.35):
        super(CosineMGN, self).__init__()

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
        # seperete branch
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
        
        # pool
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
        # cb
        self.cb_fgp1 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=guse_bnbias,cosine_loss_type=g_cosine_loss_type,scale=scale,margin=margin)
        self.cb_fgp2 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=guse_bnbias,cosine_loss_type=g_cosine_loss_type,scale=scale,margin=margin)
        self.cb_fgp3 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=guse_bnbias,cosine_loss_type=g_cosine_loss_type,scale=scale,margin=margin)
        
        self.cb_f0p2 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=puse_bnbias,cosine_loss_type=p_cosine_loss_type,scale=scale,margin=margin)
        self.cb_f1p2 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=puse_bnbias,cosine_loss_type=p_cosine_loss_type,scale=scale,margin=margin)
      
        self.cb_f0p3 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=puse_bnbias,cosine_loss_type=p_cosine_loss_type,scale=scale,margin=margin)
        self.cb_f1p3 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=puse_bnbias,cosine_loss_type=p_cosine_loss_type,scale=scale,margin=margin)
        self.cb_f2p3 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=puse_bnbias,cosine_loss_type=p_cosine_loss_type,scale=scale,margin=margin)
        
        self.p_cosine_loss_type = p_cosine_loss_type
        self.g_cosine_loss_type = g_cosine_loss_type

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x,label=None):

        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1).squeeze(dim=3).squeeze(dim=2)
        zg_p2 = self.maxpool_zg_p2(p2).squeeze(dim=3).squeeze(dim=2)
        zg_p3 = self.maxpool_zg_p3(p3).squeeze(dim=3).squeeze(dim=2)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :].squeeze(dim=3).squeeze(dim=2)
        z1_p2 = zp2[:, :, 1:2, :].squeeze(dim=3).squeeze(dim=2)

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :].squeeze(dim=3).squeeze(dim=2)
        z1_p3 = zp3[:, :, 1:2, :].squeeze(dim=3).squeeze(dim=2)
        z2_p3 = zp3[:, :, 2:3, :].squeeze(dim=3).squeeze(dim=2)

        rfgp1,fgp1,cls_fgp1 = self.cb_fgp1(zg_p1,label)
        rfgp2,fgp2,cls_fgp2 = self.cb_fgp2(zg_p2,label)
        rfgp3,fgp3,cls_fgp3 = self.cb_fgp3(zg_p3,label)

        if self.p_cosine_loss_type == '':
            label = None
        rf0p2,f0p2,cls_f0p2 = self.cb_f0p2(z0_p2,label)
        rf1p2,f1p2,cls_f1p2 = self.cb_f1p2(z1_p2,label)

        rf0p3,f0p3,cls_f0p3 = self.cb_f0p3(z0_p3,label)
        rf1p3,f1p3,cls_f1p3 = self.cb_f1p3(z1_p3,label)
        rf2p3,f2p3,cls_f2p3 = self.cb_f2p3(z2_p3,label)

        predict = torch.cat([fgp1, fgp2, fgp3, f0p2, f1p2, f0p3, f1p3, f2p3], dim=1)
        if self.training:
            if self.use_center:
                return rfgp1, rfgp2, rfgp3, \
                        cls_fgp1, cls_fgp2, cls_fgp3, cls_f0p2, cls_f1p2, cls_f0p3, cls_f1p3, cls_f2p3
            else:
                return rfgp1, rfgp2, rfgp3, cls_fgp1, cls_fgp2, cls_fgp3, cls_f0p2, cls_f1p2, cls_f0p3, cls_f1p3, cls_f2p3
        else:
            return predict


class CosineMGN2D(nn.Module):
    def __init__(self, num_classes, model_path, last_stride=1,backbone='resnet50', \
                        num_share_layer3=1,
                        pool_type='max',part_pool_type='max',\
                        use_center=False,guse_bnbias=True,puse_bnbias=True,\
                        g_cosine_loss_type='',p_cosine_loss_type='',scale=30.0,margin=0.35,
                        cb2d_bnfirst=True):
        super(CosineMGN2D, self).__init__()

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
        # seperete branch
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
        
        # pool
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

        self.pool_zp2 = pool2d_zp2
        self.pool_zp3 = pool2d_zp3
        # cb
        self.cb_fgp1 = ClassBlock2D(2048,256,num_classes,1,1,dropout=False,relu=True,pool_type=pool_type,use_bnbias=guse_bnbias,cosine_loss_type=g_cosine_loss_type,scale=scale,margin=margin,reduce_bnfirst=cb2d_bnfirst)
        self.cb_fgp2 = ClassBlock2D(2048,256,num_classes,1,1,dropout=False,relu=True,pool_type=pool_type,use_bnbias=guse_bnbias,cosine_loss_type=g_cosine_loss_type,scale=scale,margin=margin,reduce_bnfirst=cb2d_bnfirst)
        self.cb_fgp3 = ClassBlock2D(2048,256,num_classes,1,1,dropout=False,relu=True,pool_type=pool_type,use_bnbias=guse_bnbias,cosine_loss_type=g_cosine_loss_type,scale=scale,margin=margin,reduce_bnfirst=cb2d_bnfirst)
        
        self.cb_f0p2 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=puse_bnbias,cosine_loss_type=p_cosine_loss_type,scale=scale,margin=margin)
        self.cb_f1p2 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=puse_bnbias,cosine_loss_type=p_cosine_loss_type,scale=scale,margin=margin)
      
        self.cb_f0p3 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=puse_bnbias,cosine_loss_type=p_cosine_loss_type,scale=scale,margin=margin)
        self.cb_f1p3 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=puse_bnbias,cosine_loss_type=p_cosine_loss_type,scale=scale,margin=margin)
        self.cb_f2p3 = ClassBlock1D(2048,256,num_classes,dropout=False,relu=True,use_bnbias=puse_bnbias,cosine_loss_type=p_cosine_loss_type,scale=scale,margin=margin)
        
        self.p_cosine_loss_type = p_cosine_loss_type
        self.g_cosine_loss_type = g_cosine_loss_type

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x,label=None):

        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zp2 = self.pool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :].squeeze(dim=3).squeeze(dim=2)
        z1_p2 = zp2[:, :, 1:2, :].squeeze(dim=3).squeeze(dim=2)

        zp3 = self.pool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :].squeeze(dim=3).squeeze(dim=2)
        z1_p3 = zp3[:, :, 1:2, :].squeeze(dim=3).squeeze(dim=2)
        z2_p3 = zp3[:, :, 2:3, :].squeeze(dim=3).squeeze(dim=2)

        rfgp1,fgp1,cls_fgp1 = self.cb_fgp1(p1,label)
        rfgp2,fgp2,cls_fgp2 = self.cb_fgp2(p2,label)
        rfgp3,fgp3,cls_fgp3 = self.cb_fgp3(p3,label)

        if self.p_cosine_loss_type == '':
            label = None
        rf0p2,f0p2,cls_f0p2 = self.cb_f0p2(z0_p2,label)
        rf1p2,f1p2,cls_f1p2 = self.cb_f1p2(z1_p2,label)

        rf0p3,f0p3,cls_f0p3 = self.cb_f0p3(z0_p3,label)
        rf1p3,f1p3,cls_f1p3 = self.cb_f1p3(z1_p3,label)
        rf2p3,f2p3,cls_f2p3 = self.cb_f2p3(z2_p3,label)

        predict = torch.cat([fgp1, fgp2, fgp3, f0p2, f1p2, f0p3, f1p3, f2p3], dim=1)
        if self.training:
            if self.use_center:
                return rfgp1, rfgp2, rfgp3, \
                        cls_fgp1, cls_fgp2, cls_fgp3, cls_f0p2, cls_f1p2, cls_f0p3, cls_f1p3, cls_f2p3
            else:
                return rfgp1, rfgp2, rfgp3, cls_fgp1, cls_fgp2, cls_fgp3, cls_f0p2, cls_f1p2, cls_f0p3, cls_f1p3, cls_f2p3
        else:
            return predict