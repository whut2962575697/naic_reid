import os
import cv2
import copy
import glob
import numpy as np
import json
import time
import pprint
import mmcv
import functools
import pandas as pd
import argparse
import torchvision.transforms as T

import torch
import torch.nn.functional as F
from tqdm import tqdm


device = 'cuda'
def aqe_func(feat,all_feature,k2,alpha):
    sims = np.dot(feat.reshape(1,-1),all_feature.T) # 1,N
    initial_rank = np.argpartition(-sims,range(1,k2+1)) # 1,N
    weights = sims[0,initial_rank[0,:k2]].reshape((-1,1)) # k2,1
    weights = np.power(weights,alpha)
    return np.mean(all_feature[initial_rank[0,:k2],:]*weights,axis=0)
def simple_hist_predictor(image,channel=2,thres=100): #BGR; by the last channel
    hist = cv2.calcHist([image], [channel], None, [256], [0, 256]) #绘制各个通道的直方图
    if hist[0]>thres:
        return 0
    else:
        return 1
def pil_simple_hist_predictor(image,channel=0,thres=100): #RGB; by the first channel
    hist = cv2.calcHist([image], [channel], None, [256], [0, 256]) #绘制各个通道的直方图
    if hist[0]>thres:
        return 0
    else:
        return 1
def simple_hist_predictor_func(fname):
    # print(fname)
    img = cv2.imread(fname)
    return simple_hist_predictor(img,channel=2,thres=100)

def cal_mean_std(datatype,fnames):
    print("calculate dataset mean and std...")
    R_means = []
    G_means = []
    B_means = []
    R_stds = []
    G_stds = []
    B_stds = []
    with tqdm(total=len(fnames)) as pbar:
        for fname in fnames:
            im = cv2.imread(fname)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            im_R = im[:, :, 0] / 255.0
            im_G = im[:, :, 1] / 255.0
            im_B = im[:, :, 2] / 255.0
            im_R_mean = np.mean(im_R)
            im_G_mean = np.mean(im_G)
            im_B_mean = np.mean(im_B)
            im_R_std = np.std(im_R)
            im_G_std = np.std(im_G)
            im_B_std = np.std(im_B)
            R_means.append(im_R_mean)
            G_means.append(im_G_mean)
            B_means.append(im_B_mean)
            R_stds.append(im_R_std)
            G_stds.append(im_G_std)
            B_stds.append(im_B_std)

            pbar.update(1)
    a = [R_means, G_means, B_means]
    b = [R_stds, G_stds, B_stds]
    mean = [0, 0, 0]
    std = [0, 0, 0]
    mean[0] = np.mean(a[0])
    mean[1] = np.mean(a[1])
    mean[2] = np.mean(a[2])
    std[0] = np.mean(b[0])
    std[1] = np.mean(b[1])
    std[2] = np.mean(b[2])
    print('数据集{}的RGB平均值为\n[{},{},{}]'.format(datatype,mean[0], mean[1], mean[2]))
    print('数据集{}的RGB方差为\n[{},{},{}]'.format(datatype,std[0], std[1], std[2]))

if __name__ == '__main__':
    img_dir =  '/data/Dataset/PReID/dataset2/train/'
    train_imgfilenames = glob.glob(img_dir+'*.png')
    img_dir =  '/data/Dataset/PReID/dataset2/query_a/'
    query_imgfilenames = glob.glob(img_dir+'*.png')
    img_dir =  '/data/Dataset/PReID/dataset2/gallery_a/'
    gallery_imgfilenames = glob.glob(img_dir+'*.png')

    train_imgfilenames = train_imgfilenames[:len(train_imgfilenames)//2]
    query_imgfilenames = query_imgfilenames[:len(query_imgfilenames)//2]
    gallery_imgfilenames = gallery_imgfilenames[:len(gallery_imgfilenames)//2]

    print("predict train hist_label")
    train_hist_labels = mmcv.track_parallel_progress(simple_hist_predictor_func, train_imgfilenames, 6)
    train_unique_hist_labels = sorted(list(set(train_hist_labels)))
    train_sa_index = [i for i, v in enumerate(train_hist_labels) if (v == train_unique_hist_labels[0])]
    train_sb_index = [i for i, v in enumerate(train_hist_labels) if (v == train_unique_hist_labels[1])]
    train_sa_infos = [train_imgfilenames[i] for i in train_sa_index]
    train_sb_infos = [train_imgfilenames[i] for i in train_sb_index]

    print("predict query hist_label")
    query_hist_labels = mmcv.track_parallel_progress(simple_hist_predictor_func, query_imgfilenames, 6)
    query_unique_hist_labels = sorted(list(set(query_hist_labels)))
    query_sa_index = [i for i, v in enumerate(query_hist_labels) if (v == query_unique_hist_labels[0])]
    query_sb_index = [i for i, v in enumerate(query_hist_labels) if (v == query_unique_hist_labels[1])]
    query_sa_infos = [query_imgfilenames[i] for i in query_sa_index]
    query_sb_infos = [query_imgfilenames[i] for i in query_sb_index]

    print("predict gallery hist_label")
    gallery_hist_labels = mmcv.track_parallel_progress(simple_hist_predictor_func, gallery_imgfilenames, 6)
    gallery_unique_hist_labels = sorted(list(set(gallery_hist_labels)))
    gallery_sa_index = [i for i, v in enumerate(gallery_hist_labels) if (v == gallery_unique_hist_labels[0])]
    gallery_sb_index = [i for i, v in enumerate(gallery_hist_labels) if (v == gallery_unique_hist_labels[1])]
    gallery_sa_infos = [gallery_imgfilenames[i] for i in gallery_sa_index]
    gallery_sb_infos = [gallery_imgfilenames[i] for i in gallery_sb_index]

    print('train_sa: {}, train_sb: {}'.format(len(train_sa_index), len(train_sb_index)))
    print('query_sa: {}, query_sb: {}'.format(len(query_sa_index), len(query_sb_index)))
    print('gallery_sa: {}, gallery_sb: {}'.format(len(gallery_sa_index), len(gallery_sb_index)))

    cal_mean_std(datatype='train_sa',fnames=train_sa_infos)
    cal_mean_std(datatype='train_sb',fnames=train_sb_infos)

    cal_mean_std(datatype='query_sa',fnames=query_sa_infos)
    cal_mean_std(datatype='query_sb',fnames=query_sb_infos)

    cal_mean_std(datatype='gallery_sa',fnames=gallery_sa_infos)
    cal_mean_std(datatype='gallery_sb',fnames=gallery_sb_infos)


