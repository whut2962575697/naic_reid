# -*- encoding: utf-8 -*-
'''
@File    :   config.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 18:11   xin      1.0         None
'''

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NAME = 'mfn'
_C.MODEL.BACKBONE = 'resnet50_ibn_a'
_C.MODEL.LAST_STRIDE = 1
_C.MODEL.LABEL_SMOOTH = True
_C.MODEL.PRETRAIN_PATH = 'C:/Users/xin/.torch/models/r50_ibn_a.pth'
_C.MODEL.USE_DROPOUT = False
_C.MODEL.OPT_LEVEL = 'O0'
_C.MODEL.MAX_SCALE_LOSS = 2.**24
_C.MODEL.WEIGHT = ''
_C.MODEL.USE_COS = False

_C.MODEL.BASELINE = CN()
_C.MODEL.BASELINE.POOL_TYPE = 'avg' # max or avg or gem
_C.MODEL.BASELINE.TPL_WEIGHT = 1.0
_C.MODEL.BASELINE.CE_WEIGHT = 1.0
_C.MODEL.BASELINE.COSINE_LOSS_TYPE = ''
_C.MODEL.BASELINE.S = 30.0
_C.MODEL.BASELINE.M = 0.35
_C.MODEL.BASELINE.USE_BNBAIS = False
_C.MODEL.BASELINE.USE_SESTN = False

_C.MODEL.MGN = CN()
_C.MODEL.MGN.POOL_TYPE = 'max' # max or avg
_C.MODEL.MGN.PART_POOL_TYPE = 'max'
_C.MODEL.MGN.TPL_WEIGHT = 1.0
_C.MODEL.MGN.CE_WEIGHT = 2.0
_C.MODEL.MGN.NUM_SHARE_LAYER3 = 1
_C.MODEL.MGN.USE_BNBAIS = True

_C.MODEL.COSINEMGN = CN()
_C.MODEL.COSINEMGN.POOL_TYPE = 'max' # max or avg
_C.MODEL.COSINEMGN.PART_POOL_TYPE = 'max'
_C.MODEL.COSINEMGN.TPL_WEIGHT = 1.0
_C.MODEL.COSINEMGN.PCE_WEIGHT = 2.0
_C.MODEL.COSINEMGN.GCE_WEIGHT = 1.0
_C.MODEL.COSINEMGN.NUM_SHARE_LAYER3 = 1
_C.MODEL.COSINEMGN.PUSE_BNBAIS = True
_C.MODEL.COSINEMGN.GUSE_BNBAIS = True

_C.MODEL.COSINEMGN.CB2D_BNFIRST = True

_C.MODEL.COSINEMGN.GCOSINE_LOSS_TYPE = ''
_C.MODEL.COSINEMGN.PCOSINE_LOSS_TYPE = ''
_C.MODEL.COSINEMGN.S = 30.0
_C.MODEL.COSINEMGN.M = 0.35


_C.MODEL.MFN = CN()
_C.MODEL.MFN.POOL_TYPE = 'avg' # max or avg
_C.MODEL.MFN.AUX_POOL_TYPE = 'max'
_C.MODEL.MFN.AUX_SMOOTH = True
_C.MODEL.MFN.USE_EXTRA_TRIPLET = False
_C.MODEL.MFN.USE_SESTN = False

_C.MODEL.MFN.TPL_WEIGHT = 1.0
_C.MODEL.MFN.CE_WEIGHT = 1.0
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
_C.INPUT.RE_MAX_RATIO = 0.4
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.09661545, 0.18356957, 0.21322473]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.13422933, 0.14724616, 0.19259872]
_C.INPUT.USE_CJ = False
_C.INPUT.CJ_PROB = 1.1
_C.INPUT.CJ_BRIGHNESS = 0.5
_C.INPUT.CJ_CONTRAST = 0.5
_C.INPUT.CJ_SATURATION = 0.0 
_C.INPUT.CJ_HUE = 0.5

_C.INPUT.USE_CJ = False
_C.INPUT.USE_CJ = False

_C.INPUT.NORMALIZATION = True
# Value of padding size
_C.INPUT.PADDING = 10

_C.INPUT.SEPNORM = CN()
_C.INPUT.SEPNORM.USE = False
_C.INPUT.SEPNORM.PIXEL_MEAN = [[0.1164,0.1567,0.1796],[0.4556,0.5367,0.6493]]
_C.INPUT.SEPNORM.PIXEL_STD = [[0.1467,0.1497,0.1930],[0.0750,0.0613,0.1040]]
_C.INPUT.SEPNORM.TEST_PIXEL_MEAN = [[0.1164,0.1567,0.1796],[0.5410,0.4631,0.5808]]
_C.INPUT.SEPNORM.TEST_PIXEL_STD = [[0.1467,0.1497,0.1930],[0.0444,0.0821,0.0981]]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('competition')
# Root PATH to the dataset
_C.DATASETS.DATA_PATH = ''
# PATH to train set
_C.DATASETS.TRAIN_PATH = 'train'
# PATH to query set
_C.DATASETS.QUERY_PATH = 'query'
# PATH to gallery set
_C.DATASETS.GALLERY_PATH = 'gallery'
_C.DATASETS.EXEMPLAR = CN()
_C.DATASETS.EXEMPLAR.USE = False
_C.DATASETS.EXEMPLAR.PATH = 'exemplar'
_C.DATASETS.EXEMPLAR.IMS_PER_BATCH = 32

_C.DATASETS.EXEMPLAR.SOFTMAX = CN()
_C.DATASETS.EXEMPLAR.SOFTMAX.USE = False
_C.DATASETS.EXEMPLAR.SOFTMAX.WEIGHT = 0.1

_C.DATASETS.EXEMPLAR.TRIPLET = CN()
_C.DATASETS.EXEMPLAR.TRIPLET.USE = False
_C.DATASETS.EXEMPLAR.TRIPLET.WEIGHT = 0.1


_C.DATASETS.EXEMPLAR.MEMORY = CN()
_C.DATASETS.EXEMPLAR.MEMORY.USE = False
_C.DATASETS.EXEMPLAR.MEMORY.NUM_FEATS = 2048
_C.DATASETS.EXEMPLAR.MEMORY.KNN = 6
_C.DATASETS.EXEMPLAR.MEMORY.ALPHA = 0.01
_C.DATASETS.EXEMPLAR.MEMORY.BETA = 0.05
_C.DATASETS.EXEMPLAR.MEMORY.KNN_START_EPOCH = 14 # add 4 to warmup epoch
_C.DATASETS.EXEMPLAR.MEMORY.LAMBDA = 0.3

_C.DATASETS.HIST_LABEL = CN()
_C.DATASETS.HIST_LABEL.USE = False
_C.DATASETS.HIST_LABEL.LOSS_WEIGHT = 1.0
# _C.DATASETS.EXEMPLAR.NUM_FEATS = 2048
# _C.DATASETS.EXEMPLAR.KNN = 6
# _C.DATASETS.EXEMPLAR.ALPHA = 0.01
# _C.DATASETS.EXEMPLAR.BETA = 0.05
# _C.DATASETS.EXEMPLAR.KNN_START_EPOCH = 14 # add 4 to warmup epoch
# _C.DATASETS.EXEMPLAR.LAMBDA = 0.3
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.BALANCE = False
_C.DATALOADER.NUM_LEAST = -1
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.FP16 = False
_C.SOLVER.FIX_BACKBONE = False

_C.SOLVER.MAX_EPOCHS = 300

_C.SOLVER.BASE_LR = 0.03
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.MARGIN = 0.5

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [60, 120]

_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_EPOCH = 10
_C.SOLVER.WARMUP_BEGAIN_LR = 3e-4
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.EVAL_PERIOD = 1
_C.SOLVER.START_SAVE_EPOCH = 100

_C.SOLVER.TENSORBOARD = CN()
_C.SOLVER.TENSORBOARD.USE = False
_C.SOLVER.TENSORBOARD.LOG_PERIOD = 20

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 32
_C.SOLVER.SYNCBN = True

_C.SOLVER.CENTER_LOSS = CN()

_C.SOLVER.CENTER_LOSS.USE = False
_C.SOLVER.CENTER_LOSS.LR = 0.5
_C.SOLVER.CENTER_LOSS.WEIGHT = 0.0005
_C.SOLVER.CENTER_LOSS.ALPHA = 1.0

_C.SOLVER.CENTER_LOSS.OPTIMIZER_NAME = "SGD"
_C.SOLVER.CENTER_LOSS.WEIGHT_DECAY = 0.0
_C.SOLVER.CENTER_LOSS.MOMENTUM = 0.0

_C.SOLVER.CENTER_LOSS.NUM_FEATS = 2048

_C.SOLVER.MIXUP = CN()
_C.SOLVER.MIXUP.USE = False
_C.SOLVER.MIXUP.ALPHA = 0.75
_C.SOLVER.MIXUP.NEG_INSTANCE = 2
_C.SOLVER.MIXUP.CE_WEIGHT = 0.5
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 16
_C.TEST.WEIGHT = ""
_C.TEST.DEBUG = False
_C.TEST.MULTI_GPU = False
_C.TEST.CMC = [1,5,10]
_C.TEST.VIS = False
_C.TEST.VIS_Q_NUM = 10
_C.TEST.VIS_G_NUM = 5
_C.TEST.RERANK = False
_C.TEST.RANDOMPERM = 0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "E:/data/reid/output"

# Alias for easy usage
cfg = _C