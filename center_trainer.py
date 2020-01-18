# -*- encoding: utf-8 -*-
'''
@File    :   trainer.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 19:23   xin      1.0         None
'''

import torch
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import logging
from evaluate import eval_func, re_rank
from evaluate import euclidean_dist
from utils import AvgerageMeter, calculate_acc
import os.path as osp
import os
from common.sync_bn import convert_model
from common.optimizers import LRScheduler,WarmupMultiStepLR

from torch.optim import SGD
from utils.model import make_optimizer
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    import apex
except:
    pass


class CenterTrainer(object):
    def __init__(self, cfg, model, train_dl, val_dl,
                 loss_func, num_query, num_gpus):
        self.cfg = cfg
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.loss_func = loss_func
        self.num_query = num_query

        self.loss_avg = AvgerageMeter()
        self.acc_avg = AvgerageMeter()
        self.train_epoch = 1
        self.batch_cnt = 0

        self.logger = logging.getLogger('reid_baseline.train')
        self.log_period = cfg.SOLVER.LOG_PERIOD
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self.eval_period = cfg.SOLVER.EVAL_PERIOD
        self.output_dir = cfg.OUTPUT_DIR
        self.device = cfg.MODEL.DEVICE
        self.epochs = cfg.SOLVER.MAX_EPOCHS

        if cfg.SOLVER.TENSORBOARD.USE:
            summary_dir = os.path.join(cfg.OUTPUT_DIR,'summaries/')
            os.makedirs(summary_dir,exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=summary_dir)
        self.current_iteration = 0

        self.model.cuda()
        self.logger.info(self.model)

        if num_gpus > 1:
        
            self.optim = make_optimizer(self.model,opt=self.cfg.SOLVER.OPTIMIZER_NAME,lr=cfg.SOLVER.BASE_LR,weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,momentum=0.9)
            self.center_optim = make_optimizer(self.loss_func.center_criterion,opt=cfg.SOLVER.CENTER_LOSS.OPTIMIZER_NAME,\
                                    lr = cfg.SOLVER.CENTER_LOSS.LR,weight_decay=cfg.SOLVER.CENTER_LOSS.WEIGHT_DECAY,\
                                    momentum =  cfg.SOLVER.CENTER_LOSS.MOMENTUM,nesterov=False)
            self.scheduler = WarmupMultiStepLR(self.optim, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,cfg.SOLVER.WARMUP_EPOCH, cfg.SOLVER.WARMUP_METHOD)
            self.logger.info(self.optim)
            self.logger.info(self.center_optim)

            self.mix_precision = (cfg.MODEL.OPT_LEVEL != "O0")
            if self.mix_precision:
                self.model, [self.optim,self.center_optim] = amp.initialize(self.model, [self.optim,self.center_optim], opt_level=cfg.MODEL.OPT_LEVEL,max_loss_scale=cfg.MODEL.MAX_SCALE_LOSS)
                self.logger.info('Using apex for mix_precision with opt_level {}'.format(cfg.MODEL.OPT_LEVEL))

            self.model = nn.DataParallel(self.model)
            if cfg.SOLVER.SYNCBN:
                if self.mix_precision:
                    self.model = apex.parallel.convert_syncbn_model(self.model)
                    self.logger.info('More than one gpu used, convert model to use SyncBN.')
                    self.logger.info('Using apex SyncBN implementation')
                else:
                    self.model = convert_model(self.model)
                    self.model.cuda()
                    self.logger.info('More than one gpu used, convert model to use SyncBN.')
                    self.logger.info('Using pytorch SyncBN implementation')
                    self.logger.info(self.model)

            self.logger.info('Trainer Built')

            return

        else:

            self.optim = make_optimizer(self.model,opt=self.cfg.SOLVER.OPTIMIZER_NAME,lr=cfg.SOLVER.BASE_LR,weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,momentum=0.9)
            self.center_optim = make_optimizer(self.loss_func.center_criterion,opt=cfg.SOLVER.CENTER_LOSS.OPTIMIZER_NAME,\
                                                lr = cfg.SOLVER.CENTER_LOSS.LR,weight_decay=cfg.SOLVER.CENTER_LOSS.WEIGHT_DECAY,\
                                                momentum =  cfg.SOLVER.CENTER_LOSS.MOMENTUM,nesterov=False)
            self.scheduler = WarmupMultiStepLR(self.optim, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,cfg.SOLVER.WARMUP_EPOCH, cfg.SOLVER.WARMUP_METHOD)
            self.logger.info(self.optim)
            self.logger.info(self.center_optim)
            
            self.mix_precision = (cfg.MODEL.OPT_LEVEL != "O0")
            if self.mix_precision:
                self.model, [self.optim,self.center_optim] = amp.initialize(self.model, [self.optim,self.center_optim], opt_level=cfg.MODEL.OPT_LEVEL,max_loss_scale=cfg.MODEL.MAX_SCALE_LOSS)
                self.logger.info('Using apex for mix_precision with opt_level {}'.format(cfg.MODEL.OPT_LEVEL))

            return

    def handle_new_batch(self):
        if self.current_iteration % self.cfg.SOLVER.TENSORBOARD.LOG_PERIOD == 0:
            if self.summary_writer:
                self.summary_writer.add_scalar('Train/lr',self.scheduler.get_lr()[0],self.current_iteration)
                self.summary_writer.add_scalar('Train/loss',self.loss_avg.avg,self.current_iteration)
                self.summary_writer.add_scalar('Train/acc',self.acc_avg.avg,self.current_iteration)

        self.batch_cnt += 1
        self.current_iteration += 1
        if self.batch_cnt % self.cfg.SOLVER.LOG_PERIOD == 0:
            self.logger.info('Epoch[{}] Iteration[{}/{}] Loss: {:.3f},'
                             'Acc: {:.3f}, Base Lr: {:.2e}'
                             .format(self.train_epoch, self.batch_cnt,
                                     len(self.train_dl), self.loss_avg.avg,
                                     self.acc_avg.avg, self.scheduler.get_lr()[0]))

    def handle_new_epoch(self):

        self.batch_cnt = 1

        lr = self.scheduler.get_lr()[0]
        self.logger.info('Epoch {} done'.format(self.train_epoch))
        self.logger.info('-' * 20)

        torch.save(self.model.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_epoch_last.pth'))
        torch.save(self.optim.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_epoch_last_optim.pth'))

        torch.save(self.center_optim.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_epoch_last_center_optim.pth'))


        if self.train_epoch > self.cfg.SOLVER.START_SAVE_EPOCH and self.train_epoch % self.checkpoint_period == 0:
            self.save()
        if (self.train_epoch > 0 and self.train_epoch % self.eval_period == 0) or self.train_epoch == 50 :
            self.evaluate()
            pass
        self.scheduler.step()
        self.train_epoch += 1

    def step(self, batch):
        self.model.train()
        self.optim.zero_grad()
        self.center_optim.zero_grad()

        img, target = batch
        img, target = img.cuda(), target.cuda()
        if self.cfg.MODEL.USE_COS:
            outputs = self.model(img,target)
        else:
            outputs = self.model(img)


        loss,tpl,ce,ct = self.loss_func(outputs, target,in_detail=True)
        
        if self.current_iteration % self.cfg.SOLVER.TENSORBOARD.LOG_PERIOD == 0:
            if self.summary_writer:
                self.summary_writer.add_scalar('Train/tpl',tpl,self.current_iteration)
                self.summary_writer.add_scalar('Train/ce',ce,self.current_iteration)
                self.summary_writer.add_scalar('Train/ct',ct,self.current_iteration)



        if self.mix_precision:
            with amp.scale_loss(loss, self.optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optim.step()
        if self.mix_precision:
            # [todo] fix the center's step
            pass
            # for param in amp.master_params(self.center_optim):
            #     param.grad.data *= (0.5 / self.cfg.SOLVER.CENTER_LOSS.WEIGHT)
        else:
            for param in self.loss_func.center_criterion.parameters():
                # param.grad.data *= (1.0 / self.cfg.SOLVER.CENTER_LOSS.WEIGHT)
                param.grad.data *= (self.cfg.SOLVER.CENTER_LOSS.ALPHA / self.cfg.SOLVER.CENTER_LOSS.WEIGHT)

        self.center_optim.step()
        # acc = (score.max(1)[1] == target).float().mean()
        acc = calculate_acc(self.cfg, outputs, target)

        self.loss_avg.update(loss.cpu().item())
        self.acc_avg.update(acc.cpu().item())

        return self.loss_avg.avg, self.acc_avg.avg

    def evaluate(self):
        self.model.eval()
        num_query = self.num_query
        feats, pids, camids = [], [], []
        with torch.no_grad():
            for batch in tqdm(self.val_dl, total=len(self.val_dl),
                              leave=False):
                data, pid, camid, _ = batch
                data = data.cuda()

                # ff = torch.FloatTensor(data.size(0), 2048).zero_()
                # for i in range(2):
                #     if i == 1:
                #         data = data.index_select(3, torch.arange(data.size(3) - 1, -1, -1).long().to('cuda'))
                #     outputs = self.model(data)
                #     f = outputs.data.cpu()
                #     ff = ff + f

                ff = self.model(data).data.cpu()
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

                feats.append(ff)
                pids.append(pid)
                camids.append(camid)
        feats = torch.cat(feats, dim=0)
        pids = torch.cat(pids, dim=0)
        camids = torch.cat(camids, dim=0)
        if self.cfg.TEST.RANDOMPERM <=0 :
            query_feat = feats[:num_query]
            query_pid = pids[:num_query]
            query_camid = camids[:num_query]

            gallery_feat = feats[num_query:]
            gallery_pid = pids[num_query:]
            gallery_camid = camids[num_query:]

            distmat = euclidean_dist(query_feat, gallery_feat)


            cmc, mAP, _ = eval_func(distmat.numpy(), query_pid.numpy(), gallery_pid.numpy(),
                                    query_camid.numpy(), gallery_camid.numpy(),
                                    )
        else:
            cmc = 0
            mAP = 0
            seed = torch.random.get_rng_state()
            torch.manual_seed(0)
            for i in range(self.cfg.TEST.RANDOMPERM):
                index = torch.randperm(feats.size()[0])
                # print(index[:10])
                query_feat = feats[index][:num_query]
                query_pid = pids[index][:num_query]
                query_camid = camids[index][:num_query]

                gallery_feat = feats[index][num_query:]
                gallery_pid = pids[index][num_query:]
                gallery_camid = camids[index][num_query:]

                distmat = euclidean_dist(query_feat, gallery_feat)

                _cmc, _mAP, _ = eval_func(distmat.numpy(), query_pid.numpy(), gallery_pid.numpy(),
                                        query_camid.numpy(), gallery_camid.numpy(),
                                        )
                cmc += _cmc/self.cfg.TEST.RANDOMPERM
                mAP += _mAP/self.cfg.TEST.RANDOMPERM
            torch.random.set_rng_state(seed)

        self.logger.info('Validation Result:')
        self.logger.info('mAP: {:.2%}'.format(mAP))
        for r in self.cfg.TEST.CMC:
            self.logger.info('CMC Rank-{}: {:.2%}'.format(r, cmc[r - 1]))
        
        self.logger.info('average of mAP and rank1: {:.2%}'.format((mAP+cmc[0])/2.0))
        
        self.logger.info('-' * 20)

        if self.summary_writer:
            self.summary_writer.add_scalar('Valid/rank1',cmc[0],self.train_epoch)
            self.summary_writer.add_scalar('Valid/mAP',mAP,self.train_epoch)
            self.summary_writer.add_scalar('Valid/rank1_mAP',(mAP+cmc[0])/2.0,self.train_epoch)


    def save(self):
        torch.save(self.model.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_epoch' + str(self.train_epoch) + '.pth'))
        torch.save(self.optim.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_epoch' + str(
                                                         self.train_epoch) + '_optim.pth'))
        torch.save(self.center_optim.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_epoch' + str(
                                                         self.train_epoch) + '_center_optim.pth'))





