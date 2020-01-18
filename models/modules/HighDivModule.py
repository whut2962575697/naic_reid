# -*- encoding: utf-8 -*-
'''
@File    :   HighDivModule.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/7 10:32   xin      1.0         None
'''
from torch import nn
import torch.nn.functional as F

class HighDivModule(nn.Module):
    def __init__(self, in_channels, order=1):
        super(HighDivModule, self).__init__()
        self.order = order
        self.inter_channels = in_channels // 8 * 2
        for j in range(self.order):
            for i in range(j + 1):
                name = 'order' + str(self.order) + '_' + str(j + 1) + '_' + str(i + 1)
                setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, self.inter_channels, 1, padding=0, bias=False))
                        )
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i + 1)
            setattr(self, name, nn.Sequential(nn.Conv2d(self.inter_channels, in_channels, 1, padding=0, bias=False),
                                              nn.Sigmoid()
                                              )
                    )

    def forward(self, x):
        y = []
        for j in range(self.order):
            for i in range(j + 1):
                name = 'order' + str(self.order) + '_' + str(j + 1) + '_' + str(i + 1)
                layer = getattr(self, name)
                y.append(layer(x))
        y_ = []
        cnt = 0
        for j in range(self.order):
            y_temp = 1
            for i in range(j + 1):
                y_temp = y_temp * y[cnt]
                cnt += 1
            y_.append(F.relu(y_temp))

        # y_ = F.relu(y_)
        y__ = 0
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i + 1)
            layer = getattr(self, name)
            y__ += layer(y_[i])
        out = x * y__ / self.order
        return out  # , y__/ self.order