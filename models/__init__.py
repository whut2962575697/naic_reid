# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 18:10   xin      1.0         None
'''

from .baseline import Baseline
from .cosine_baseline import CosineBaseline
from .mgn import MGN
from .mfn import MFN
from .pcb import PCB
from .small_mhn_pcb import MHN_smallPCB
from .mgn_bnneck import MGNBNNeck
from .cosine_mgn import CosineMGN,CosineMGN2D
def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == "baseline":
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH,
                         backbone=cfg.MODEL.BACKBONE,pool_type=cfg.MODEL.BASELINE.POOL_TYPE, use_dropout=cfg.MODEL.USE_DROPOUT)
    elif cfg.MODEL.NAME == "cosine_baseline":
        model = CosineBaseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH,
                                backbone=cfg.MODEL.BACKBONE,pool_type=cfg.MODEL.BASELINE.POOL_TYPE, use_dropout=cfg.MODEL.USE_DROPOUT,\
                                cosine_loss_type=cfg.MODEL.BASELINE.COSINE_LOSS_TYPE,s=cfg.MODEL.BASELINE.S,m=cfg.MODEL.BASELINE.M,use_bnbias=cfg.MODEL.BASELINE.USE_BNBAIS,use_sestn=cfg.MODEL.BASELINE.USE_SESTN )
    elif cfg.MODEL.NAME == "mgn":
        model = MGN(num_classes, cfg.MODEL.PRETRAIN_PATH,last_stride=cfg.MODEL.LAST_STRIDE, backbone=cfg.MODEL.BACKBONE,pool_type=cfg.MODEL.MGN.POOL_TYPE,part_pool_type=cfg.MODEL.MGN.PART_POOL_TYPE,\
                    use_center=cfg.SOLVER.CENTER_LOSS.USE,\
                    num_share_layer3 = cfg.MODEL.MGN.NUM_SHARE_LAYER3,\
                    use_bnbias=cfg.MODEL.MGN.USE_BNBAIS)
    elif cfg.MODEL.NAME == "cosinemgn":
        model = CosineMGN(num_classes, cfg.MODEL.PRETRAIN_PATH,last_stride=cfg.MODEL.LAST_STRIDE, backbone=cfg.MODEL.BACKBONE,num_share_layer3 = cfg.MODEL.COSINEMGN.NUM_SHARE_LAYER3,\
                    pool_type=cfg.MODEL.COSINEMGN.POOL_TYPE,part_pool_type=cfg.MODEL.COSINEMGN.PART_POOL_TYPE,\
                    use_center=cfg.SOLVER.CENTER_LOSS.USE,\
                    guse_bnbias=cfg.MODEL.COSINEMGN.GUSE_BNBAIS,puse_bnbias=cfg.MODEL.COSINEMGN.PUSE_BNBAIS,\
                    g_cosine_loss_type=cfg.MODEL.COSINEMGN.GCOSINE_LOSS_TYPE,p_cosine_loss_type=cfg.MODEL.COSINEMGN.PCOSINE_LOSS_TYPE,\
                    scale=cfg.MODEL.COSINEMGN.S,margin=cfg.MODEL.COSINEMGN.M)
    elif cfg.MODEL.NAME == "cosinemgn2d":
        model = CosineMGN2D(num_classes, cfg.MODEL.PRETRAIN_PATH,last_stride=cfg.MODEL.LAST_STRIDE, backbone=cfg.MODEL.BACKBONE,num_share_layer3 = cfg.MODEL.COSINEMGN.NUM_SHARE_LAYER3,\
                    pool_type=cfg.MODEL.COSINEMGN.POOL_TYPE,part_pool_type=cfg.MODEL.COSINEMGN.PART_POOL_TYPE,\
                    use_center=cfg.SOLVER.CENTER_LOSS.USE,\
                    guse_bnbias=cfg.MODEL.COSINEMGN.GUSE_BNBAIS,puse_bnbias=cfg.MODEL.COSINEMGN.PUSE_BNBAIS, \
                    cb2d_bnfirst=cfg.MODEL.COSINEMGN.CB2D_BNFIRST, \
                    g_cosine_loss_type=cfg.MODEL.COSINEMGN.GCOSINE_LOSS_TYPE,p_cosine_loss_type=cfg.MODEL.COSINEMGN.PCOSINE_LOSS_TYPE,\
                    scale=cfg.MODEL.COSINEMGN.S,margin=cfg.MODEL.COSINEMGN.M)

    elif cfg.MODEL.NAME == "mfn":
        model = MFN(num_classes, cfg.MODEL.PRETRAIN_PATH,last_stride=cfg.MODEL.LAST_STRIDE, backbone=cfg.MODEL.BACKBONE,pool_type=cfg.MODEL.MFN.POOL_TYPE,aux_pool_type=cfg.MODEL.MFN.AUX_POOL_TYPE,aux_smooth=cfg.MODEL.MFN.AUX_SMOOTH,\
                    use_histlabel = cfg.DATASETS.HIST_LABEL.USE, \
                    use_extra_triplet = cfg.MODEL.MFN.USE_EXTRA_TRIPLET, \
                    use_sestn = cfg.MODEL.MFN.USE_SESTN
                    )
    elif cfg.MODEL.NAME == "pcb":
        model = PCB(num_classes, cfg.MODEL.PRETRAIN_PATH, backbone=cfg.MODEL.BACKBONE)
    elif cfg.MODEL.NAME == "small_mhn_pcb":
        model = MHN_smallPCB(num_classes, cfg.MODEL.PRETRAIN_PATH, backbone=cfg.MODEL.BACKBONE)
    elif cfg.MODEL.NAME == "mgn_bnneck":
        model = MGNBNNeck(num_classes, cfg.MODEL.PRETRAIN_PATH,last_stride=cfg.MODEL.LAST_STRIDE, backbone=cfg.MODEL.BACKBONE,pool_type=cfg.MODEL.MGN.POOL_TYPE,\
                    use_center=cfg.SOLVER.CENTER_LOSS.USE,\
                    num_share_layer3 = cfg.MODEL.MGN.NUM_SHARE_LAYER3)
    else:
        model = None
    return model

