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
    
    origin_path = root_dir+'dataset2/'+'pid_dataset/'
    pseudo_path = '../rep_work_dirs/testb_pseudo_hist_065_080/'

    save_dir = root_dir+'testb_pseudo_hist_065_080_dataset/'
    os.makedirs(save_dir,exist_ok=True)

    # trainVal
    print('==> copy origin to rep trainVal')
    trainVal_path = save_dir + 'rep_trainVal/'
    trainVal2_path = save_dir + 'rep_trainVal2/'

    os.makedirs(trainVal_path,exist_ok=True)
    os.makedirs(trainVal2_path,exist_ok=True)

    rep_pids = os.listdir(origin_path)
    with tqdm(total = len(rep_pids)) as pbar:
        for pid in rep_pids:
            imgs = os.listdir(os.path.join(origin_path, pid))
            for img in imgs:
                shutil.copy(os.path.join(os.path.join(origin_path, pid), img), os.path.join(trainVal_path, img))
            if len(imgs)>=2:
                for img in imgs:
                    shutil.copy(os.path.join(os.path.join(origin_path, pid), img), os.path.join(trainVal2_path, img))
            pbar.update(1)
    print('==> copy pseduo to rep trainVal')
    
    rep_pids = os.listdir(pseudo_path)
    with tqdm(total = len(rep_pids)) as pbar:
        for pid in rep_pids:
            imgs = os.listdir(os.path.join(pseudo_path, pid))
            for img in imgs:
                shutil.copy(os.path.join(os.path.join(pseudo_path, pid), img), os.path.join(trainVal_path, img))
            if len(imgs)>=2:
                for img in imgs:
                    shutil.copy(os.path.join(os.path.join(pseudo_path, pid), img), os.path.join(trainVal2_path, img))
            pbar.update(1)
