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
import glob

from dataset.data import read_image
from skimage.io import imsave, imread
from PIL import ImageFile, Image
from tqdm import tqdm

def process_dataset(txt_label, root_path, save_path,camera_start_id=0):
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
                new_filename = pid+"_c"+str(i+camera_start_id)+".png"
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
    rep_dir = root_dir+'dataset2/'

    save_dir = root_dir+'rep_dataset/'
    os.makedirs(save_dir,exist_ok=True)

    if 0:
        print('==> copy rep to pid dataset')
        process_dataset(rep_dir+'fix_train_list.txt',  rep_dir+'train/',  save_dir+'pid_dataset/')

    # fix seed
    kfold = 0
    
    # load pre and rep id seperately
    rep_train = pd.read_csv(rep_dir+'fix_train_list.txt',sep=' ',header=None,names=['filename','identity'])
    rep_train['filename'] = rep_train['filename'].apply(lambda x: x.split('/')[1]) 

    root_path = save_dir+'pid_dataset/'

    # deal with rep data only

    rep_pids = rep_train['identity'].astype(str).tolist()
    rep_pids = list(set(rep_pids))
    rep_pids = sorted(rep_pids)

    if 0:
        # trainVal
        print('==> copy to rep trainVal')
        trainVal_path = save_dir + 'rep_trainVal/'
        trainVal2_path = save_dir + 'rep_trainVal2/'

        os.makedirs(trainVal_path,exist_ok=True)
        os.makedirs(trainVal2_path,exist_ok=True)

        with tqdm(total = len(rep_pids)) as pbar:
            for pid in rep_pids:
                imgs = os.listdir(os.path.join(root_path, pid))
                for img in imgs:
                    shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(trainVal_path, img))
                if len(imgs)>=2:
                    for img in imgs:
                        shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(trainVal2_path, img))
                pbar.update(1)

    if 1:
        np.random.seed(kfold)
        # split train and val(from rep only)
        np.random.shuffle(rep_pids)
        train_rep_pids = rep_pids[:int(len(rep_pids)*0.85)]
        val_rep_pids = rep_pids[int(len(rep_pids)*0.85):]
        print(train_rep_pids[:10])
        # # query gallery
        # print('==> copy to rep query and gallery')
        # for q_fold in range(5):
        #     query_path = save_dir+'rep_f{}_query_{}/'.format(kfold,q_fold)
        #     gallery_path = save_dir + 'rep_f{}_gallery_{}/'.format(kfold,q_fold)
        #     os.makedirs(query_path,exist_ok=True)
        #     os.makedirs(gallery_path,exist_ok=True)
        
        #     with tqdm(total = len(val_rep_pids)) as pbar:

        #         img_id = 0
        #         for pid in val_rep_pids:
        #             imgs = os.listdir(os.path.join(root_path, pid))
        #             imgs = sorted(imgs)
        #             np.random.shuffle(imgs)
        #             for img in imgs:
        #                 img_id+=1

        #                 if (img_id+q_fold) % 5 == 0:
        #                     shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(query_path, img))
        #                 else:
        #                     shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(gallery_path, img))
        #             pbar.update(1)

        
        # query gallery
        print('==> copy to rep query and gallery')
        query_path = save_dir+'rep_f{}_query/'.format(kfold)
        gallery_path = save_dir + 'rep_f{}_gallery/'.format(kfold)
        os.makedirs(query_path,exist_ok=True)
        os.makedirs(gallery_path,exist_ok=True)
    
        with tqdm(total = len(val_rep_pids)) as pbar:

            img_id = 0
            for pid in val_rep_pids:
                imgs = os.listdir(os.path.join(root_path, pid))
                imgs = sorted(imgs)
                np.random.shuffle(imgs)
                for img in imgs:
                    img_id+=1
                    if img_id % 5 == 0:
                    # if img_id % 13 == 0:
                        shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(query_path, img))
                    else:
                        shutil.copy(os.path.join(os.path.join(root_path, pid), img), os.path.join(gallery_path, img))
                pbar.update(1)

        print('==> copy to rep train')
        train_path = save_dir + 'rep_f{}_train/'.format(kfold)
        train2_path = save_dir + 'rep_f{}_train2/'.format(kfold)
        train4_path = save_dir + 'rep_f{}_train4/'.format(kfold)

        os.makedirs(train_path,exist_ok=True)
        os.makedirs(train2_path,exist_ok=True)
        os.makedirs(train4_path,exist_ok=True)
        with tqdm(total = len(train_rep_pids)) as pbar:
            for pid in train_rep_pids:
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