# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 18:10   xin      1.0         None
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .triplet import TripletLoss, CrossEntropyLabelSmooth,NegMixupTripletLoss
from .advdiv_loss import AdvDivLoss
from .center_loss import CenterLoss
from .arcloss import ArcCos

class BaseLineLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes, margin=0.3,label_smooth=True,tpl_weight=1.0,ce_weight=1.0):
        super(BaseLineLoss, self).__init__()
        self.label_smooth = label_smooth
        self.num_classes = num_classes
        self.margin = margin
        self.tpl_weight = tpl_weight
        self.ce_weight = ce_weight

    def forward(self, outputs, labels,in_detail=False):
        score, feat = outputs
        if self.label_smooth:
            cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        else:
            cross_entropy_loss = F.cross_entropy
        if self.tpl_weight>0.0:
            triplet_loss = TripletLoss(self.margin)
            Triplet_Loss = triplet_loss(feat, labels)

            CrossEntropy_Loss = cross_entropy_loss(score, labels)
            # loss_sum = Triplet_Loss + CrossEntropy_Loss
            loss_sum = self.tpl_weight*Triplet_Loss + self.ce_weight * CrossEntropy_Loss

            print('\r[loss] total:%.2f\t tpl:%.4f\t ce:%.2f\t' % (
                loss_sum.data.cpu().numpy(),
                Triplet_Loss.data.cpu().numpy(),
                CrossEntropy_Loss.data.cpu().numpy()),
                end=' ')
 
        else:
            CrossEntropy_Loss = cross_entropy_loss(score, labels)
            # loss_sum = Triplet_Loss + CrossEntropy_Loss
            Triplet_Loss = 0.0
            loss_sum = self.ce_weight * CrossEntropy_Loss

            print('\r[loss] total:%.2f\t tpl:%.4f\t ce:%.2f\t' % (
                loss_sum.data.cpu().numpy(),
                0.0,
                CrossEntropy_Loss.data.cpu().numpy()),
                end=' ')
        if in_detail:
            return loss_sum,Triplet_Loss,CrossEntropy_Loss
        return loss_sum

class MGNLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes, margin=0.3,label_smooth=True,tpl_weight=1.0,ce_weight=2.0):
        super(MGNLoss, self).__init__()
        self.label_smooth = label_smooth
        self.num_classes = num_classes
        self.margin = margin
        self.tpl_weight = tpl_weight
        self.ce_weight = ce_weight
    def forward(self, outputs, labels,in_detail=False):
        if self.label_smooth:
            cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        else:
            cross_entropy_loss = F.cross_entropy
        triplet_loss = TripletLoss(self.margin)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[:3]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[3:]]
        # drop loss
        # selected_ids = np.random.choice(5,4,replace=False)
        # selected_ids += 3 # keep global 
        # selected_ids = list(range(3))+list(selected_ids) 
        # print(selected_ids)


        # CrossEntropy_Loss = [cross_entropy_loss(output, labels) for idx,output in enumerate(outputs[3:]) if idx in selected_ids]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        # loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss
        loss_sum = self.tpl_weight*Triplet_Loss + self.ce_weight * CrossEntropy_Loss


        print('\r[loss] total:%.2f\t tpl:%.4f\t ce:%.2f\t' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        if in_detail:
            return loss_sum,self.tpl_weight*Triplet_Loss,self.ce_weight * CrossEntropy_Loss
        return loss_sum

class CosineMGNLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes, margin=0.3,label_smooth=True,tpl_weight=1.0,pce_weight=1.0,gce_weight=1.0):
        super(CosineMGNLoss, self).__init__()
        self.label_smooth = label_smooth
        self.num_classes = num_classes
        self.margin = margin
        self.tpl_weight = tpl_weight
        self.pce_weight = pce_weight
        self.gce_weight = gce_weight
    def forward(self, outputs, labels,in_detail=False):
        if self.label_smooth:
            cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        else:
            cross_entropy_loss = F.cross_entropy
        triplet_loss = TripletLoss(self.margin)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[:3]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        GCrossEntropy_Loss = [F.cross_entropy(output, labels) for output in outputs[3:6]]
        GCrossEntropy_Loss = sum(GCrossEntropy_Loss) / len(GCrossEntropy_Loss)

        PCrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[6:]]
        PCrossEntropy_Loss = sum(PCrossEntropy_Loss) / len(PCrossEntropy_Loss)

        loss_sum = self.tpl_weight * Triplet_Loss + self.gce_weight * GCrossEntropy_Loss + self.pce_weight * PCrossEntropy_Loss

        print('\r[loss] total:%.2f\t tpl:%.4f\t pce:%.2f\t gce:%.2f\t' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            PCrossEntropy_Loss.data.cpu().numpy(),
            GCrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        if in_detail:
            return loss_sum, Triplet_Loss, PCrossEntropy_Loss, GCrossEntropy_Loss
        return loss_sum

class MFNLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes, margin=0.6,label_smooth=True):
        super(MFNLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.label_smooth = label_smooth
    def forward(self, outputs, labels,in_detail=False):

        if self.label_smooth:
            cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        else:
            cross_entropy_loss = F.cross_entropy

        triplet_loss = TripletLoss(self.margin)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[4:]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[:4]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)


        loss_sum = Triplet_Loss + CrossEntropy_Loss
        print('\r[loss] total:%.2f\t tpl:%.4f\t ce:%.2f\t' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        if in_detail:
            return loss_sum, Triplet_Loss,CrossEntropy_Loss
        return loss_sum

class MFNHistLabelLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes, margin=0.6,label_smooth=True,hl_weight=1.0):
        super(MFNHistLabelLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.label_smooth = label_smooth
        self.hl_weight = hl_weight
    def forward(self, outputs, labels,hist_label,in_detail=False):

        if self.label_smooth:
            cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        else:
            cross_entropy_loss = F.cross_entropy

        triplet_loss = TripletLoss(self.margin)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[4:8]]
        # print(len(Triplet_Loss))
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[:4]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        HistLabel_Loss = F.cross_entropy(outputs[8], hist_label) 

        loss_sum = Triplet_Loss + CrossEntropy_Loss + self.hl_weight*HistLabel_Loss
        print('\r[loss] total:%.2f\t tpl:%.4f\t ce:%.2f\t hlce:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy(),
            HistLabel_Loss.data.cpu().numpy()),
              end=' ')
        if in_detail:
            return loss_sum, Triplet_Loss,CrossEntropy_Loss,HistLabel_Loss
        return loss_sum

class PCBLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes,label_smooth=True):
        super(PCBLoss, self).__init__()
        self.num_classes = num_classes
        self.label_smooth = label_smooth


    def forward(self, outputs, labels,in_detail=False):

        if self.label_smooth:
            cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        else:
            cross_entropy_loss = F.cross_entropy

        _, _, _, logits_list, _ = outputs

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in logits_list]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = CrossEntropy_Loss

        print('\r[loss] total:%.2f\t ce:%.2f\t' % (
            loss_sum.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        if in_detail:
            return loss_sum, 0.0,CrossEntropy_Loss
        return loss_sum
class MHNPCBLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes, alpha=1.0,label_smooth=True):
        super(MHNPCBLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha


    def forward(self, outputs, labels):

        if self.label_smooth:
            cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        else:
            cross_entropy_loss = F.cross_entropy
        logits_list, fea = outputs
        adv_div_loss = AdvDivLoss(len(fea))

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in logits_list]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        AdvDiv_Loss = [adv_div_loss(output) for output in fea]
        AdvDiv_Loss = sum(AdvDiv_Loss) / len(AdvDiv_Loss)

        loss_sum = CrossEntropy_Loss + AdvDiv_Loss*self.alpha

        print('\r[loss] total:%.2f\t advdiv:%.2f\t ce:%.2f\t' % (
            loss_sum.data.cpu().numpy(),
            AdvDiv_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        return loss_sum

# centerloss
class BaseLineCenterLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes,num_feats, margin=0.3,label_smooth=True,tpl_weight=1.0,ce_weight=1.0,center_weight=0.0005):
        super(BaseLineCenterLoss, self).__init__()
        self.label_smooth = label_smooth
        self.num_classes = num_classes
        self.margin = margin
        self.tpl_weight = tpl_weight
        self.ce_weight = ce_weight

        self.center_weight = center_weight
        self.num_feats = num_feats
        self.center_criterion = CenterLoss(num_classes=num_classes, feat_dim=num_feats, use_gpu=True)  # center loss


    def forward(self, outputs, labels,in_detail=False):
        score, feat = outputs
        if self.label_smooth:
            cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        else:
            cross_entropy_loss = F.cross_entropy
        triplet_loss = TripletLoss(self.margin)
        Triplet_Loss = triplet_loss(feat, labels)
        CrossEntropy_Loss = cross_entropy_loss(score, labels)
        
        Center_Loss = self.center_criterion(feat,labels)
        # loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss
        loss_sum = self.tpl_weight*Triplet_Loss + self.ce_weight * CrossEntropy_Loss + self.center_weight * Center_Loss

        print('\r[loss] total:%.2f\t tpl:%.4f\t ce:%.2f\tct:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy(),
            Center_Loss.data.cpu().numpy()),
              end=' ')
        if in_detail:
            return loss_sum,Triplet_Loss,CrossEntropy_Loss,Center_Loss
        return loss_sum

class MGNCenterLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes,num_feats, margin=0.3,label_smooth=True,tpl_weight=1.0,ce_weight=2.0,center_weight=0.0005):
        super(MGNCenterLoss, self).__init__()
        self.label_smooth = label_smooth
        self.num_classes = num_classes
        self.margin = margin
        self.tpl_weight = tpl_weight
        self.ce_weight = ce_weight

        self.center_weight = center_weight
        self.num_feats = num_feats
        self.center_criterion = CenterLoss(num_classes=num_classes, feat_dim=num_feats, use_gpu=True)  # center loss

    def forward(self, outputs, labels,in_detail=False):
        if self.label_smooth:
            cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        else:
            cross_entropy_loss = F.cross_entropy
        triplet_loss = TripletLoss(self.margin)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[:3]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[3:11]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        feats = torch.cat([torch.cat(outputs[:3],dim=1),torch.cat(outputs[11:],dim=1)],dim=1)
        # import pdb;pdb.set_trace()
        Center_Loss = self.center_criterion(feats,labels)
        # loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss
        loss_sum = self.tpl_weight*Triplet_Loss + self.ce_weight * CrossEntropy_Loss +self.center_weight * Center_Loss


        print('\r[loss] total:%.2f\t tpl:%.4f\t ce:%.2f\t ct:%.2f\t' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy(),Center_Loss.data.cpu().numpy()),
              end=' ')
        if in_detail:
            return loss_sum, Triplet_Loss,CrossEntropy_Loss,Center_Loss
        return loss_sum

class MFNCenterLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes,num_feats, margin=0.5,label_smooth=True,tpl_weight=1.0,ce_weight=1.0,center_weight=0.0005):
        super(MFNCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.label_smooth = label_smooth

        self.tpl_weight = tpl_weight
        self.ce_weight = ce_weight

        self.center_weight = center_weight
        self.num_feats = num_feats
        self.center_criterion = CenterLoss(num_classes=num_classes, feat_dim=num_feats, use_gpu=True)  # center loss
    def forward(self, outputs, labels):

        if self.label_smooth:
            cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        else:
            cross_entropy_loss = F.cross_entropy

        triplet_loss = TripletLoss(self.margin)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[4:]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[:4]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        feats = torch.cat(outputs[4:],dim=1)
        # import pdb;pdb.set_trace()
        Center_Loss = self.center_criterion(feats,labels)
        # loss_sum = Triplet_Loss + 1 * CrossEntropy_Loss
        loss_sum = self.tpl_weight*Triplet_Loss + self.ce_weight * CrossEntropy_Loss +self.center_weight * Center_Loss


        print('\r[loss] total:%.2f\t tpl:%.4f\t ce:%.2f\t ct:%.2f\t' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy(),Center_Loss.data.cpu().numpy()),
              end=' ')
        return loss_sum
# Negative Mixup loss

class MFNNegMixupLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes, margin=0.6,label_smooth=True,K1=4,K2=4,tpl_weight=1.0,ce_weight=1.0,mixup_ce_weight=0.5):
        super(MFNNegMixupLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.label_smooth = label_smooth
        self.K1 = K1
        self.K2 = K2
        self.tpl_weight = tpl_weight
        self.ce_weight = ce_weight
        self.mixup_ce_weight = mixup_ce_weight
    def forward(self, outputs, labels,mx_outputs,mx_target1,mx_target2,lamb):

        if self.label_smooth:
            cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        else:
            cross_entropy_loss = F.cross_entropy
        # mixup tpl
        triplet_loss = NegMixupTripletLoss(self.margin,self.K1,self.K2)

        Triplet_Loss = [triplet_loss(output, labels,mx_outputs[i+4]) for i,output in enumerate(outputs[4:])]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[:4]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        # mixup ce
        if self.mixup_ce_weight>0.0:
            Mixup_CrossEntropy_Loss = [lamb*cross_entropy_loss(output, mx_target1)+(1-lamb)*cross_entropy_loss(output, mx_target2) for output in mx_outputs[:4]]
            Mixup_CrossEntropy_Loss = sum(Mixup_CrossEntropy_Loss) / len(Mixup_CrossEntropy_Loss)

            loss_sum = self.tpl_weight*Triplet_Loss + self.ce_weight*CrossEntropy_Loss+self.mixup_ce_weight*Mixup_CrossEntropy_Loss
            print('\r[loss] total:%.2f\t tpl:%.4f\t ce:%.2f\t mixce:%.2f\t' % (
                loss_sum.data.cpu().numpy(),
                Triplet_Loss.data.cpu().numpy(),
                CrossEntropy_Loss.data.cpu().numpy(),
                Mixup_CrossEntropy_Loss.data.cpu().numpy()),
                  end=' ')
        else:

            loss_sum = self.tpl_weight*Triplet_Loss + self.ce_weight*CrossEntropy_Loss
            print('\r[loss] total:%.2f\t tpl:%.4f\t ce:%.2f\t ' % (
                loss_sum.data.cpu().numpy(),
                Triplet_Loss.data.cpu().numpy(),
                CrossEntropy_Loss.data.cpu().numpy()),
                  end=' ')
        return loss_sum
class MFNUnknownIdentityLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes, margin=0.6,label_smooth=True,exemplar_softmax_weight=0.1,exemplar_triplet_weight=0.0):
        super(MFNUnknownIdentityLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.label_smooth = label_smooth
        self.exemplar_softmax_weight = exemplar_softmax_weight
    def forward(self, outputs, labels):

        if self.label_smooth:
            cross_entropy_loss = CrossEntropyLabelSmooth(self.num_classes)
        else:
            cross_entropy_loss = F.cross_entropy

        triplet_loss = TripletLoss(self.margin)

        Triplet_Loss = [triplet_loss(output[:labels.size()[0]], labels) for output in outputs[4:]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output[:labels.size()[0]], labels) for output in outputs[:4]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        # unlabel loss
        # UnknownIdentityRejection_Loss = [-F.log_softmax(output[labels.size()[0]:],dim=1).mean(0).sum() for output in outputs[:4]]
        UnknownIdentityRejection_Loss = [-F.log_softmax(output[labels.size()[0]:],dim=1).mean() for output in outputs[:4]]

        UnknownIdentityRejection_Loss = sum(UnknownIdentityRejection_Loss) / len(UnknownIdentityRejection_Loss)
        loss_sum = Triplet_Loss + CrossEntropy_Loss+self.exemplar_softmax_weight*UnknownIdentityRejection_Loss
        print('\r[loss] total:%.2f\t tpl:%.4f\t ce:%.2f\t uirl:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy(),
            UnknownIdentityRejection_Loss.data.cpu().numpy()
            ),
              end=' ')
        return loss_sum
def make_loss(cfg, num_classes):
    if cfg.SOLVER.CENTER_LOSS.USE == True:
        if cfg.MODEL.NAME in ["mgn","mgn_bnneck"]:
            loss = MGNCenterLoss(num_classes,cfg.SOLVER.CENTER_LOSS.NUM_FEATS,cfg.SOLVER.MARGIN,label_smooth=cfg.MODEL.LABEL_SMOOTH,tpl_weight=cfg.MODEL.MGN.TPL_WEIGHT,ce_weight=cfg.MODEL.MGN.CE_WEIGHT,center_weight=cfg.SOLVER.CENTER_LOSS.WEIGHT)
        elif cfg.MODEL.NAME == "mfn":
            loss = MFNCenterLoss(num_classes,cfg.SOLVER.CENTER_LOSS.NUM_FEATS,cfg.SOLVER.MARGIN,label_smooth=cfg.MODEL.LABEL_SMOOTH,tpl_weight=cfg.MODEL.MFN.TPL_WEIGHT,ce_weight=cfg.MODEL.MFN.CE_WEIGHT,center_weight=cfg.SOLVER.CENTER_LOSS.WEIGHT)
        elif cfg.MODEL.NAME in ["baseline","cosine_baseline"]:
            loss = BaseLineCenterLoss(num_classes,cfg.SOLVER.CENTER_LOSS.NUM_FEATS,cfg.SOLVER.MARGIN,label_smooth=cfg.MODEL.LABEL_SMOOTH,tpl_weight=cfg.MODEL.BASELINE.TPL_WEIGHT,ce_weight=cfg.MODEL.BASELINE.CE_WEIGHT,center_weight=cfg.SOLVER.CENTER_LOSS.WEIGHT)
        else:
            loss = None
    else:
        if cfg.SOLVER.MIXUP.USE:
            if cfg.MODEL.NAME == "mfn":
                loss= MFNNegMixupLoss(num_classes,cfg.SOLVER.MARGIN,label_smooth=cfg.MODEL.LABEL_SMOOTH,\
                    K1=cfg.DATALOADER.NUM_INSTANCE,K2=cfg.SOLVER.MIXUP.NEG_INSTANCE,\
                    tpl_weight=cfg.MODEL.MFN.TPL_WEIGHT,ce_weight=cfg.MODEL.MFN.CE_WEIGHT,\
                    mixup_ce_weight = cfg.SOLVER.MIXUP.CE_WEIGHT
                    )
            else:
                loss = None
        else:
            if cfg.DATASETS.EXEMPLAR.SOFTMAX.USE or cfg.DATASETS.EXEMPLAR.TRIPLET.USE:
                if cfg.MODEL.NAME == "mfn":
                    loss= MFNUnknownIdentityLoss(num_classes,cfg.SOLVER.MARGIN,label_smooth=cfg.MODEL.LABEL_SMOOTH,\
                                                exemplar_softmax_weight=cfg.DATASETS.EXEMPLAR.SOFTMAX.WEIGHT,\
                                                exemplar_triplet_weight=cfg.DATASETS.EXEMPLAR.TRIPLET.WEIGHT)
            if cfg.DATASETS.HIST_LABEL.USE :
                if cfg.MODEL.NAME == "mfn":
                    loss= MFNHistLabelLoss(num_classes,cfg.SOLVER.MARGIN,label_smooth=cfg.MODEL.LABEL_SMOOTH,hl_weight=cfg.DATASETS.HIST_LABEL.LOSS_WEIGHT)
            else:
                if cfg.MODEL.NAME in ["cosinemgn","cosinemgn2d"]:
                    loss = CosineMGNLoss(num_classes,cfg.SOLVER.MARGIN,label_smooth=cfg.MODEL.LABEL_SMOOTH,\
                                        tpl_weight=cfg.MODEL.COSINEMGN.TPL_WEIGHT,pce_weight=cfg.MODEL.COSINEMGN.PCE_WEIGHT,gce_weight=cfg.MODEL.COSINEMGN.GCE_WEIGHT)
                elif cfg.MODEL.NAME in ["baseline","cosine_baseline"]:
                    if cfg.DATALOADER.SAMPLER == 'softmax':
                        tpl_weight = 0.0
                    else:
                        tpl_weight = cfg.MODEL.BASELINE.TPL_WEIGHT
                    loss = BaseLineLoss(num_classes,cfg.SOLVER.MARGIN,label_smooth=cfg.MODEL.LABEL_SMOOTH,tpl_weight=tpl_weight,ce_weight=cfg.MODEL.BASELINE.CE_WEIGHT)
                elif cfg.MODEL.NAME in ["mgn","mgn_bnneck"]:
                    loss = MGNLoss(num_classes,cfg.SOLVER.MARGIN,label_smooth=cfg.MODEL.LABEL_SMOOTH,tpl_weight=cfg.MODEL.MGN.TPL_WEIGHT,ce_weight=cfg.MODEL.MGN.CE_WEIGHT)
                elif cfg.MODEL.NAME in ["mfn"]:
                    loss= MFNLoss(num_classes,cfg.SOLVER.MARGIN,label_smooth=cfg.MODEL.LABEL_SMOOTH)
                elif cfg.MODEL.NAME == "pcb":
                    loss = PCBLoss(num_classes,label_smooth=cfg.MODEL.LABEL_SMOOTH)
                elif cfg.MODEL.NAME == "small_mhn_pcb":
                    loss = MHNPCBLoss(num_classes,label_smooth=cfg.MODEL.LABEL_SMOOTH)
                else:
                    loss = None
    return loss

