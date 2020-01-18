# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 18:11   xin      1.0         None
'''

import os
import torch
import argparse

from config import cfg
from utils import setup_logger
from dataset import make_dataloader,make_sepnorm_dataloader
from models import build_model
from losses import make_loss
from trainer import BaseTrainer
from center_trainer import CenterTrainer
from negMixup_trainer import NegMixupTrainer
from exemplarMemoryTrainer import ExemplarMemoryTrainer
from unknownIdentityTrainer import UIRLTrainer
from histLabelTrainer import HistLabelTrainer

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_gpus = torch.cuda.device_count()
    logger = setup_logger('reid_baseline', output_dir, 0)
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info('Running with config:\n{}'.format(cfg))
    if cfg.INPUT.SEPNORM.USE:
        train_dl, val_dl, num_query, num_classes = make_sepnorm_dataloader(cfg, num_gpus)
    elif cfg.DATASETS.EXEMPLAR.USE:
        train_dl, val_dl, num_query, num_classes,exemplar_dl = make_dataloader(cfg, num_gpus)
    else:
        train_dl, val_dl, num_query, num_classes = make_dataloader(cfg, num_gpus)

    model = build_model(cfg, num_classes)
    loss = make_loss(cfg, num_classes)
    if cfg.SOLVER.CENTER_LOSS.USE == True:
        trainer = CenterTrainer(cfg, model, train_dl, val_dl,
                      loss, num_query, num_gpus)
    else:
        if cfg.SOLVER.MIXUP.USE:
            trainer = NegMixupTrainer(cfg, model, train_dl, val_dl,
                              loss, num_query, num_gpus)
        elif cfg.DATASETS.EXEMPLAR.USE:
            if cfg.DATASETS.EXEMPLAR.MEMORY.USE:
                trainer = ExemplarMemoryTrainer(cfg, model, train_dl, val_dl,exemplar_dl,
                                  loss, num_query, num_gpus)
            else:
                trainer = UIRLTrainer(cfg, model, train_dl, val_dl,exemplar_dl,
                                  loss, num_query, num_gpus)
        elif cfg.DATASETS.HIST_LABEL.USE:
            trainer = HistLabelTrainer(cfg, model, train_dl, val_dl,
                    loss, num_query, num_gpus)
        else:
            trainer = BaseTrainer(cfg, model, train_dl, val_dl,
                              loss, num_query, num_gpus)
    if cfg.INPUT.SEPNORM.USE:
        logger.info('train transform0: \n{}'.format(train_dl.dataset.transform0))
        logger.info('train transform1: \n{}'.format(train_dl.dataset.transform1))

        logger.info('valid transform0: \n{}'.format(val_dl.dataset.transform0))
        logger.info('valid transform1: \n{}'.format(val_dl.dataset.transform1))

    else:
        logger.info('train transform: \n{}'.format(train_dl.dataset.transform))
        logger.info('valid transform: \n{}'.format(val_dl.dataset.transform))
    logger.info(type(model))
    logger.info(loss)
    logger.info(trainer)
    for epoch in range(trainer.epochs):
        for batch in trainer.train_dl:
            trainer.step(batch)
            trainer.handle_new_batch()
        trainer.handle_new_epoch()


if __name__ == "__main__":
    main()

