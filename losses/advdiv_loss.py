# -*- encoding: utf-8 -*-
'''
@File    :   advdiv_loss.py
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/7 10:45   xin      1.0         None
'''


import torch.nn as nn
import torch.nn.functional as F



############# Adv network ####
def Adv_hook(module, grad_in, grad_out):
    return ((grad_in[0] * (-1), grad_in[1]))


class AdvDivLoss(nn.Module):
    """
    Attention AdvDiverse Loss
    x : is the vector
    """

    def __init__(self, parts=4):
        super(AdvDivLoss, self).__init__()
        self.parts = parts
        self.fc_pre = nn.Sequential(nn.Linear(256, 128, bias=False))
        self.fc = nn.Sequential(nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 128), nn.BatchNorm1d(128))
        self.fc_pre.register_backward_hook(Adv_hook)

    def forward(self, x):
        x = nn.functional.normalize(x)
        x = self.fc_pre(x)
        x = self.fc(x)
        x = nn.functional.normalize(x)
        out = 0
        num = int(x.size(0) / self.parts)
        for i in range(self.parts):
            for j in range(self.parts):
                if i < j:
                    out += (
                        (x[i * num:(i + 1) * num, :] - x[j * num:(j + 1) * num, :]).norm(dim=1, keepdim=True)).mean()
        return out * 2 / (self.parts * (self.parts - 1))