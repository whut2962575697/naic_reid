import numpy as np
import os
import shutil
import torch

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))

# def make_optimizer(model,opt, lr,weight_decay,initial_lr,momentum=0.9):
#     # params = []
#     # for key, value in model.named_parameters():
#     #     params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
#     # if opt == 'SGD':
#     #     optimizer = getattr(torch.optim, opt)(params, momentum=momentum,nesterov=True)
#     # elif opt == 'AMSGRAD':
#     #     optimizer = getattr(torch.optim,'Adam')(params,amsgrad=True)
#     # else:
#     #     optimizer = getattr(torch.optim, opt)(params)
#     # return optimizer
#     if opt == 'SGD':
#         optimizer = getattr(torch.optim, opt)([{'params': model.parameters(), 'initial_lr': initial_lr}],lr=lr,weight_decay=weight_decay, momentum=momentum,nesterov=True)
#     elif opt == 'AMSGRAD':
#         optimizer = getattr(torch.optim,'Adam')([{'params': model.parameters(), 'initial_lr': initial_lr}],lr=lr,weight_decay=weight_decay,amsgrad=True)
#     else:
#         optimizer = getattr(torch.optim, opt)([{'params': model.parameters(), 'initial_lr': initial_lr}],lr=lr,weight_decay=weight_decay)

#     return optimizer
def make_optimizer(model,opt, lr,weight_decay,momentum=0.9,nesterov=True):
    if opt == 'SGD':
        optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum,nesterov=nesterov)
    elif opt == 'AMSGRAD':
        optimizer = getattr(torch.optim,'Adam')(model.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=True)
    else:
        optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay)
    return optimizer

def make_optimizer_partial(weights,opt, lr,weight_decay,momentum=0.9,nesterov=True):
    if opt == 'SGD':
        optimizer = getattr(torch.optim, opt)(weights,lr=lr,weight_decay=weight_decay, momentum=momentum,nesterov=nesterov)
    elif opt == 'AMSGRAD':
        optimizer = getattr(torch.optim,'Adam')(weights,lr=lr,weight_decay=weight_decay,amsgrad=True)
    else:
        optimizer = getattr(torch.optim, opt)(weights,lr=lr,weight_decay=weight_decay)
    return optimizer


class AvgerageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res
