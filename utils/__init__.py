# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 19:15   xin      1.0         None
'''

from .utils import norm, denorm, Visualizer
from .model import accuracy, count_parameters_in_MB
from .model import AvgerageMeter
from .logging import setup_logger
from .file_op import check_isfile, mkdir_if_missing, create_exp_dir
from .vistools import rank_list_to_im


def calculate_acc(cfg, outputs, labels):
    if cfg.MODEL.NAME in ["baseline","cosine_baseline"]:
        acc = (outputs[0].max(1)[1] == labels).float().mean()
    elif cfg.MODEL.NAME in ["mgn","mgn_bnneck",'cosinemgn','cosinemgn2d']:
        acc = 0.
        for score in outputs[3:11]:
            _acc = (score.max(1)[1] == labels).float().mean()
            acc += _acc
        acc = acc / 8.
    elif cfg.MODEL.NAME in ["mfn"]:
        acc = 0.
        for score in outputs[:4]:
            _acc = (score[:labels.size()[0]].max(1)[1] == labels).float().mean()
            acc += _acc
        acc = acc / 4.
    elif cfg.MODEL.NAME == "pcb":
        acc = 0.
        _, _, _, logits_list, _ = outputs
        for score in logits_list:
            _acc = (score.max(1)[1] == labels).float().mean()
            acc += _acc
        acc = acc / (1.0*len(logits_list))
    else:
        acc = None
    return acc