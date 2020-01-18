# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 18:10   xin      1.0         None
'''

from .transform import RandomErasing
from .collate_batch import train_collate_fn,train_histlabel_collate_fn
from .collate_batch import val_collate_fn,val_histlabel_collate_fn
from .triplet_sampler import RandomIdentitySampler,RandomIdentityBalanceSampler
from .data import ImageDataset, init_dataset,SepNormImageDataset
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader


def get_trm(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = [
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        ]
        if cfg.INPUT.USE_CJ:
            if cfg.INPUT.CJ_PROB>1.0:
                transform.append(T.ColorJitter(brightness=cfg.INPUT.CJ_BRIGHNESS, contrast=cfg.INPUT.CJ_CONTRAST,saturation=cfg.INPUT.CJ_SATURATION, hue=cfg.INPUT.CJ_HUE))
            else:
                transform.append(T.RandomApply([T.ColorJitter(brightness=cfg.INPUT.CJ_BRIGHNESS, contrast=cfg.INPUT.CJ_CONTRAST,saturation=cfg.INPUT.CJ_SATURATION, hue=cfg.INPUT.CJ_HUE)],p=cfg.INPUT.CJ_PROB))

        transform.append(T.ToTensor())
        if cfg.INPUT.NORMALIZATION:
            transform.append(normalize_transform)
        transform.append(RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN,sh=cfg.INPUT.RE_MAX_RATIO))
    else:
        transform = [
            T.Resize(cfg.INPUT.SIZE_TEST),
            # T.ColorJitter(brightness=cfg.INPUT.CJ_BRIGHNESS, contrast=cfg.INPUT.CJ_CONTRAST,saturation=cfg.INPUT.CJ_SATURATION, hue=cfg.INPUT.CJ_HUE),
            T.ToTensor(),
            # normalize_transform
        ]
        if cfg.INPUT.NORMALIZATION:
            transform.append(normalize_transform)

    transform = T.Compose(transform)
    return transform


# def make_dataloader(cfg, num_gpus=1):
#     train_trm = get_trm(cfg, is_train=True)
#     val_trm = get_trm(cfg, is_train=False)

#     num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus
#     dataset = init_dataset(cfg)

#     num_classes = dataset.num_train_pids
#     train_set = ImageDataset(dataset.train, cfg, train_trm)
#     if cfg.DATALOADER.SAMPLER == 'softmax':
#         train_loader = DataLoader(
#             train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH * num_gpus, shuffle=True,
#             num_workers=num_workers,
#             collate_fn=train_collate_fn
#         )
#     else:
#         train_loader = DataLoader(
#             train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH * num_gpus,
#             sampler=RandomIdentitySampler(dataset.train,
#                 cfg.SOLVER.IMS_PER_BATCH * num_gpus,
#                 cfg.DATALOADER.NUM_INSTANCE * num_gpus),
#             num_workers=num_workers, collate_fn=train_collate_fn
#         )

#     val_set = ImageDataset(dataset.query + dataset.gallery, cfg, val_trm)
#     val_loader = DataLoader(
#         val_set, batch_size=cfg.TEST.IMS_PER_BATCH * num_gpus, shuffle=False,
#         num_workers=num_workers,
#         collate_fn=val_collate_fn
#     )
#     return train_loader, val_loader, len(dataset.query), num_classes


def make_dataloader(cfg, num_gpus=1):
    train_trm = get_trm(cfg, is_train=True)
    val_trm = get_trm(cfg, is_train=False)
    exemplar_trm = get_trm(cfg, is_train=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus
    dataset = init_dataset(cfg)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, cfg, train_trm)
    if cfg.DATASETS.HIST_LABEL.USE:
        train_cf = train_histlabel_collate_fn
        valid_cf = val_histlabel_collate_fn
    else:
        train_cf = train_collate_fn
        valid_cf = val_collate_fn

    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True,
            num_workers=num_workers,
            collate_fn=train_cf
        )
    else:
        if cfg.DATALOADER.BALANCE == False:
            sampler=RandomIdentitySampler(dataset.train,
                    cfg.SOLVER.IMS_PER_BATCH,
                    cfg.DATALOADER.NUM_INSTANCE )
        else:
            sampler=RandomIdentityBalanceSampler(dataset.train,
                    cfg.SOLVER.IMS_PER_BATCH,
                    cfg.DATALOADER.NUM_INSTANCE,
                    num_least=cfg.DATALOADER.NUM_LEAST )
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler= sampler,
            num_workers=num_workers, collate_fn=train_cf
        )
    val_set = ImageDataset(dataset.query + dataset.gallery, cfg, val_trm)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=num_workers,
        collate_fn=valid_cf
    )
    # exemplar
    if cfg.DATASETS.EXEMPLAR.USE:
        exemplar_set = ImageDataset(dataset.exemplar, cfg, exemplar_trm)
        exemplar_loader = DataLoader(
            exemplar_set, batch_size=cfg.DATASETS.EXEMPLAR.IMS_PER_BATCH, shuffle=False,
            num_workers=num_workers,
            collate_fn=train_cf
        )
        return train_loader, val_loader, len(dataset.query), num_classes, exemplar_loader
    return train_loader, val_loader, len(dataset.query), num_classes

def get_trm_by_meanstd(cfg,mean,std, is_train=True):
    normalize_transform = T.Normalize(mean=mean, std=std)
    if is_train:
        transform = [
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            # normalize_transform,
        ]
        if cfg.INPUT.NORMALIZATION:
            transform.append(normalize_transform)
        transform.append(RandomErasing(probability=cfg.INPUT.RE_PROB, mean=mean,sh=cfg.INPUT.RE_MAX_RATIO))
    else:
        transform = [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            # normalize_transform
        ]
        if cfg.INPUT.NORMALIZATION:
            transform.append(normalize_transform)

    transform = T.Compose(transform)
    return transform
def make_sepnorm_dataloader(cfg, num_gpus=1):
    train_trm0 = get_trm_by_meanstd(cfg,mean=cfg.INPUT.SEPNORM.PIXEL_MEAN[0], std=cfg.INPUT.SEPNORM.PIXEL_STD[0], is_train=True)
    train_trm1 = get_trm_by_meanstd(cfg,mean=cfg.INPUT.SEPNORM.PIXEL_MEAN[1], std=cfg.INPUT.SEPNORM.PIXEL_STD[1], is_train=True)

    val_trm0 = get_trm_by_meanstd(cfg,mean=cfg.INPUT.SEPNORM.PIXEL_MEAN[0], std=cfg.INPUT.SEPNORM.PIXEL_STD[0], is_train=False)
    val_trm1 = get_trm_by_meanstd(cfg,mean=cfg.INPUT.SEPNORM.PIXEL_MEAN[1], std=cfg.INPUT.SEPNORM.PIXEL_STD[1], is_train=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus
    dataset = init_dataset(cfg)

    num_classes = dataset.num_train_pids
    train_set = SepNormImageDataset(dataset.train, cfg, train_trm0,train_trm1)
    if cfg.DATASETS.HIST_LABEL.USE:
        train_cf = train_histlabel_collate_fn
        valid_cf = val_histlabel_collate_fn
    else:
        train_cf = train_collate_fn
        valid_cf = val_collate_fn

    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True,
            num_workers=num_workers,
            collate_fn=train_cf
        )
    else:
        if cfg.DATALOADER.BALANCE == False:
            sampler=RandomIdentitySampler(dataset.train,
                    cfg.SOLVER.IMS_PER_BATCH,
                    cfg.DATALOADER.NUM_INSTANCE )
        else:
            sampler=RandomIdentityBalanceSampler(dataset.train,
                    cfg.SOLVER.IMS_PER_BATCH,
                    cfg.DATALOADER.NUM_INSTANCE,
                    num_least=cfg.DATALOADER.NUM_LEAST )
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler= sampler,
            num_workers=num_workers, collate_fn=train_cf
        )

    val_set = SepNormImageDataset(dataset.query + dataset.gallery, cfg, val_trm0,val_trm1)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=num_workers,
        collate_fn=valid_cf
    )

    return train_loader, val_loader, len(dataset.query), num_classes

