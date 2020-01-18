# -*- encoding: utf-8 -*-
'''
@File    :   filter_dataset.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/21 18:57   xin      1.0         None
'''

import os
import torch

import torchvision.transforms as T
from models.baseline import Baseline
from evaluate import eval_func, euclidean_dist, re_rank
import numpy as np

from config import cfg
from common.sync_bn import convert_model
from models import build_model
from dataset.data import read_image
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(cfg, 1955)
para_dict = torch.load(r'E:\data\reid\exp\mfn\lr=0.03_warmup2_m=0.5/mfn_epoch226.pth')
model = torch.nn.DataParallel(model)
model = convert_model(model)
model.load_state_dict(para_dict)

transform = T.Compose([
    T.Resize((256, 128)),

    T.ToTensor(),
    # T.Normalize(mean=[0.09661545, 0.18356957, 0.21322473], std=[0.13422933, 0.14724616, 0.19259872])
])

pids = os.listdir(r'E:\data\reid\gan\train')
batch_size = 4

for pid, p_path in zip(pids, [os.path.join(r'E:\data\reid\gan\train', x) for x in pids]):
    print(pid)
    img_list = list()
    imgs = os.listdir(p_path)
    for img in [os.path.join(p_path, x) for x in imgs]:
        img = read_image(img)
        img = transform(img)
        img_list.append(img)
    img_data = torch.Tensor([t.numpy() for t in img_list])
    model = model.to(device)
    model.eval()
    iter_n = len(img_list) // batch_size
    if len(img_list) % batch_size != 0:
        iter_n += 1
    all_feature = list()
    print(len(img_list))
    if len(img_list) == 1:
        continue
    for i in range(iter_n):
        # print("batch ----%d----" % (i))

        batch_data = img_data[i * batch_size:(i + 1) * batch_size]
        with torch.no_grad():
            # batch_feature = model(batch_data).detach().cpu()

            ff = torch.FloatTensor(batch_data.size(0), 2048).zero_()
            for i in range(2):
                if i == 1:
                    batch_data = batch_data.index_select(3, torch.arange(batch_data.size(3) - 1, -1, -1).long())
                outputs = model(batch_data)
                f = outputs.data.cpu()
                ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            all_feature.append(ff)
    all_feature = torch.cat(all_feature)
    gallery_feat = all_feature
    query_feat = all_feature
    distmat = euclidean_dist(query_feat, gallery_feat).numpy()
    for m in range(distmat.shape[0]):
        for n in range(distmat.shape[1]):
            v = distmat[m, n]
            if v > 0.5:
                print(imgs[m], imgs[n])
