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

if __name__ == "__main__":
    root_dir = '/data/Dataset/PReID/'
    np.random.seed(491001)
    save_dir = root_dir+'pre/'
    root_path = save_dir+'all_dataset/'
    trainVal_path = save_dir + 'trainVal/'
    train_path = save_dir + 'train/'
    query_path = save_dir+'query/'
    gallery_path = save_dir + 'gallery/'

    exemplar_valid_path = save_dir + 'exemplar_valid/'
    os.makedirs(root_path,exist_ok=True)
    os.makedirs(trainVal_path,exist_ok=True)
    os.makedirs(train_path,exist_ok=True)
    os.makedirs(query_path,exist_ok=True)
    os.makedirs(gallery_path,exist_ok=True)
    os.makedirs(exemplar_valid_path,exist_ok=True)


    pids = os.listdir(query_path)
    for pid in pids:
        shutil.copy(os.path.join(query_path, pid), os.path.join(exemplar_valid_path, pid))

    pids = os.listdir(gallery_path)
    for pid in pids:
        shutil.copy(os.path.join(gallery_path, pid), os.path.join(exemplar_valid_path, pid))