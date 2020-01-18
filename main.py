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

from config import cfg
from utils import setup_logger
from dataset import make_dataloader
from models import build_model
from losses import make_loss
from trainer import BaseTrainer

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


def main():
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_gpus = torch.cuda.device_count()
    logger = setup_logger('reid_baseline', output_dir, 0)
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info('Running with config:\n{}'.format(cfg))
    train_dl, val_dl, num_query, num_classes = make_dataloader(cfg, num_gpus)
    model = build_model(cfg, num_classes)
    loss = make_loss(cfg, num_classes)
    trainer = BaseTrainer(cfg, model, train_dl, val_dl,
                          loss, num_query, num_gpus)
    for epoch in range(trainer.epochs):
        for batch in trainer.train_dl:
            trainer.step(batch)
            trainer.handle_new_batch()
        trainer.handle_new_epoch()


if __name__ == "__main__":
    main()

