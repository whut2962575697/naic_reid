import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# import .functional as LF
from .functional import spoc,mac,gem,adaptive_gem2d
# https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/cirtorch/layers/functional.py
# --------------------------------------
# Pooling layers
# --------------------------------------

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1], -1)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MAC(nn.Module):

    def __init__(self):
        super(MAC, self).__init__()

    def forward(self, x):
        # return LF.mac(x)
        return mac(x)
        

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SPoC(nn.Module):

    def __init__(self):
        super(SPoC, self).__init__()

    def forward(self, x):
        # return LF.spoc(x)
        return spoc(x)


    def __repr__(self):
        return self.__class__.__name__ + '()'


class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps
        self.freeze_p = freeze_p
    def forward(self, x):
        # return LF.gem(x, p=self.p, eps=self.eps)
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + \
               ', ' + 'freeze_p=' + str(self.freeze_p) +\
               ')'


class AdaptiveGeM2d(nn.Module):

    def __init__(self, output_size=(1,1),p=3.0, eps=1e-6, freeze_p=True):
        super(AdaptiveGeM2d, self).__init__()
        self.output_size = output_size
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps
        self.freeze_p = freeze_p
    
    def forward(self, x):
        # return LF.gem(x, p=self.p, eps=self.eps)
        return adaptive_gem2d(x, self.output_size,p=self.p, eps=self.eps)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'output_size='+'{}'.format(self.output_size) + \
               ','+'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + \
               ', ' + 'freeze_p=' + str(self.freeze_p) +\
               ')'

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.shape[2:]
        return F.avg_pool2d(input=x, kernel_size=(h, w))

    @staticmethod
    def out_features(in_features):
        return in_features


class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.shape[2:]
        return F.max_pool2d(input=x, kernel_size=(h, w))

    @staticmethod
    def out_features(in_features):
        return in_features


class GlobalConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.max = GlobalMaxPool2d()

    def forward(self, x):
        return torch.cat([self.avg(x), self.max(x)], 1)

    @staticmethod
    def out_features(in_features):
        return in_features * 2


class GlobalAttnPool2d(nn.Module):
    def __init__(self, in_features, activation_fn="sigmoid"):
        super().__init__()
        self.in_features = in_features
        self.activation_fn = activation_fn
        # activation_fn = MODULES.get_if_str(activation_fn)
        if activation_fn == "sigmoid":
            activation_fn = nn.Sigmoid
        else:
            raise NotImplementedError()
        self.attn = nn.Sequential(
            nn.Conv2d(
                in_features, 1, kernel_size=1, stride=1, padding=0, bias=False
            ), activation_fn()
        )

    def forward(self, x):
        x_a = self.attn(x)
        x = x * x_a
        x = torch.sum(x, dim=[-2, -1], keepdim=True)
        return x

    @staticmethod
    def out_features(in_features):
        return in_features
    
    def __repr__(self):
        return self.__class__.__name__ +\
               '(' + 'in_features=' + '{}'.format(self.in_features) +\
               ', ' + 'activation_fn=' + self.activation_fn + ')'

class GlobalAvgAttnPool2d(nn.Module):
    def __init__(self, in_features, activation_fn="sigmoid"):
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.attn = GlobalAttnPool2d(in_features, activation_fn)

    def forward(self, x):
        return torch.cat([self.avg(x), self.attn(x)], 1)

    @staticmethod
    def out_features(in_features):
        return in_features * 2


class GlobalMaxAttnPool2d(nn.Module):
    def __init__(self, in_features, activation_fn="sigmoid"):
        super().__init__()
        self.max = GlobalMaxPool2d()
        self.attn = GlobalAttnPool2d(in_features, activation_fn)

    def forward(self, x):
        return torch.cat([self.max(x), self.attn(x)], 1)

    @staticmethod
    def out_features(in_features):
        return in_features * 2


class GlobalConcatAttnPool2d(nn.Module):
    def __init__(self, in_features, activation_fn="sigmoid"):
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.max = GlobalMaxPool2d()
        self.attn = GlobalAttnPool2d(in_features, activation_fn)

    def forward(self, x):
        return torch.cat([self.avg(x), self.max(x), self.attn(x)], 1)

    @staticmethod
    def out_features(in_features):
        return in_features * 3

class GlobalAvgMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.max = GlobalMaxPool2d()

    def forward(self, x):
        return (self.avg(x)+self.max(x))/2.0

    @staticmethod
    def out_features(in_features):
        return in_features 