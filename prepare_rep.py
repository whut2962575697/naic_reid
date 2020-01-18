# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data.py
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/28 16:11   xin      1.0         None
'''
import os
import shutil
import pandas as pd
import numpy as np
import random

from dataset.data import read_image
from skimage.io import imsave, imread
from PIL import ImageFile, Image
from tqdm import tqdm

def process_dataset(txt_label, root_path, save_path):
    os.makedirs(save_path,exist_ok=True)
    with open(txt_label, 'r') as f:
        lines = f.readlines()
        with tqdm(total = len(lines)) as pbar:
            for i, line in enumerate(lines):
                data = line.split(" ")
                image_name = data[0].split("/")[1]
                pid = data[1].strip("\n")
                if not os.path.exists(os.path.join(save_path, pid)):
                    os.mkdir(os.path.join(save_path, pid))
                new_filename = pid+"_c"+str(i)+".png"
                shutil.copy(os.path.join(root_path, image_name), os.path.join(os.path.join(save_path, pid), new_filename))
                pbar.update(1)

def dataset_analyse(root_path):
    pids = os.listdir(root_path)
    counts = list()
    for pid in pids:
        imgs = os.listdir(os.path.join(root_path, pid))
        counts.append(len(imgs))
    columns = [u'pid', u'count']
    save_df = pd.DataFrame({u'pid': pids, u'count': counts},
                           columns=columns)
    save_df.to_csv('dataset_analyse.csv')


def split_dataset(root_path, train_path, query_path, gallery):
    pids = os.listdir(root_path)
    for pid in pids:
        imgs = os.listdir(os.path.join(root_path, pid))
        for img in imgs:
            shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train_path, img1))

def mk_pseudo_data(root_path, txt_label, csv_data, save_path):
    query_dic = dict()
    with open(txt_label, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.split(" ")
            image_name = data[0].split("/")[1]
            pid = data[1].strip("\n")
            query_dic[image_name] = pid
    pseudo_csv_data = pd.read_csv(csv_data)
    for query_file, gallery_file in zip(pseudo_csv_data['q_imgs'], pseudo_csv_data['g_imgs']):
        pid = query_dic[query_file]
        # if not os.path.exists(os.path.join(save_path, pid)):
        #     os.mkdir(os.path.join(save_path, pid))
        new_query_filename = pid + "_c" + query_file
        shutil.copy(os.path.join(root_path, query_file), os.path.join(save_path, new_query_filename))
        new_gallery_filename = pid + "_c" + gallery_file
        shutil.copy(os.path.join(root_path, gallery_file), os.path.join(save_path, new_gallery_filename))


if __name__ == "__main__":
    root_dir = '/data/Dataset/PReID/'
    
    save_dir = root_dir+'dataset2/'
    if 1:
        print('==> copy to pid dataset')
        process_dataset(save_dir+'fix_train_list.txt',  save_dir+'train/',  save_dir+'pid_dataset/')
    root_path = save_dir+'pid_dataset/'
    trainVal_path = save_dir + 'trainVal/'

    kfold = 1
    np.random.seed(kfold)
    
    train_path = save_dir + 'f{}_train/'.format(kfold)
    train2_path = save_dir + 'f{}_train2/'.format(kfold)
    train4_path = save_dir + 'f{}_train4/'.format(kfold)

    os.makedirs(root_path,exist_ok=True)
    os.makedirs(trainVal_path,exist_ok=True)
    os.makedirs(train_path,exist_ok=True)
    os.makedirs(train2_path,exist_ok=True)
    os.makedirs(train4_path,exist_ok=True)


    pids = os.listdir(root_path)
    pids = sorted(pids)
    # trainVal
    if 1:
        print('==> copy to trainVal')
        with tqdm(total = len(pids)) as pbar:
            for pid in pids:
                imgs = os.listdir(os.path.join(root_path, pid))
                for img in imgs:
                    shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(trainVal_path, img))
                pbar.update(1)
    # train
    np.random.shuffle(pids)
    train_pids = pids[:int(len(pids)*0.85)]
    val_pids = pids[int(len(pids)*0.85):]
    print('==> copy to train')
    with tqdm(total = len(train_pids)) as pbar:

        for pid in train_pids:
            imgs = os.listdir(os.path.join(root_path, pid))
            if 1:
                for img in imgs:
                    shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train_path, img))
            if len(imgs)>=2:
                for img in imgs:
                    shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train2_path, img))
            if len(imgs)>=4:
                for img in imgs:
                    shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(train4_path, img))
            pbar.update(1)
    print('==> copy to query and gallery')
    # query gallery
    for q_fold in range(5):
        query_path = save_dir+'f{}_query_{}/'.format(kfold,q_fold)
        gallery_path = save_dir + 'f{}_gallery_{}/'.format(kfold,q_fold)
        os.makedirs(query_path,exist_ok=True)
        os.makedirs(gallery_path,exist_ok=True)
    
        with tqdm(total = len(val_pids)) as pbar:

            img_id = 0
            for pid in val_pids:
                imgs = os.listdir(os.path.join(root_path, pid))
                imgs = sorted(imgs)
                np.random.shuffle(imgs)
                for img in imgs:
                    img_id+=1

                    if (img_id+q_fold) % 5 == 0:
                        shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(query_path, img))
                    else:
                        shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(gallery_path, img))
                pbar.update(1)
