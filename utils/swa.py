import os
import torch
from tqdm import tqdm
# https://github.com/timgaripov/swa

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.reset_running_stats()

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(model,loader,cumulative =True):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param model: model being update
        :param loader: train dataset loader for buffers average estimation.
        :param cumulative: cumulative moving average or exponential moving average
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    model.apply(reset_bn)

    if cumulative:
        momenta = {}
        model.apply(lambda module: _get_momenta(module, momenta))
        for module in momenta.keys():
            module.momentum = None

    # import pdb;pdb.set_trace()
    with tqdm(total=len(loader),leave=False) as pbar:
    # with tqdm(total=len(loader)) as pbar:
        for t,samples in enumerate(loader):
            if isinstance(samples,tuple) or isinstance(samples,list):
                input = samples[0].cuda()
            else:
                input = samples.cuda()
            input_var = torch.autograd.Variable(input)
            # import pdb;pdb.set_trace()
            model(input_var)
            pbar.update(1)

def specific_bn_update(model,loader,cumulative =True,bn_keys=[]):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param model: model being update
        :param loader: train dataset loader for buffers average estimation.
        :param cumulative: cumulative moving average or exponential moving average
        :return: None
    """
    if not check_bn(model):
        return

    # def reset_bn_func(module):
    #     if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    #         import pdb;pdb.set_trace()
    #         if module.__name__ in bn_keys:
    #             module.train()
    #             module.reset_running_stats()
    #             if cumulative:
    #                 module.momentum = None
    # model.apply(reset_bn_func)
    if isinstance(model, torch.nn.DataParallel):
        for _,_m in model.module.named_modules():
            # print([n for n,_ in _m.named_modules()])
            for n,m in _m.named_modules():
                # import pdb;pdb.set_trace()
                # if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):

                if n in bn_keys:
                    print('[bn_before]:',n,'\t',m)
                    m.train()
                    m.reset_running_stats()
                    if cumulative:
                        m.momentum = None
                    print('[bn_update]:',n,'\t',m)
                    
    # import pdb;pdb.set_trace()
    with tqdm(total=len(loader),leave=False) as pbar:
    # with tqdm(total=len(loader)) as pbar:
        for t,samples in enumerate(loader):
            if isinstance(samples,tuple) or isinstance(samples,list):
                input = samples[0].cuda()
            else:
                input = samples.cuda()
            input_var = torch.autograd.Variable(input)
            # import pdb;pdb.set_trace()
            model(input_var)
            pbar.update(1)