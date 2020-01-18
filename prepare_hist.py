import cv2
import glob
import mmcv
import numpy as np 
import pandas as pd
def simple_hist_predictor(image,channel=2,thres=100): #BGR; by the last channel
    hist = cv2.calcHist([image], [channel], None, [256], [0, 256]) #绘制各个通道的直方图
    if hist[0]>thres:
        return 0
    else:
        return 1
if __name__ == "__main__":
    if 0:
        root_dir = '/data/Dataset/PReID/dataset2/'
        df_train = pd.read_csv(root_dir+'/fix_train_list.txt',sep=' ',header=None,names=['filename','identity'])
        df_train['filename'] = df_train['filename'].apply(lambda x: x.split('/')[1]) 
        df_train['filename'] = df_train['filename'].apply(lambda x: root_dir+'train/'+x) 
        # df_train_partial = df_train.head(100)
        # df_train_partial = df_train.head(10000)
        df_train_partial = df_train


        def func(fname):
            img = cv2.imread(fname)
            return simple_hist_predictor(img,channel=2,thres=100)
        labels = mmcv.track_parallel_progress(func, df_train_partial['filename'], 8)
        df_train_partial['label'] = labels
        print("==> distribution for hist class")
        print(df_train_partial['label'].value_counts())
        df_id_cnt = df_train_partial.groupby(df_train_partial['identity']).apply(lambda x: len(np.unique(x['label']))).reset_index().rename(columns={0:'cnt'})
        print("==> distribution for the number of each identity")
        print(df_id_cnt['cnt'].value_counts())
        print("total identity:",len(set(df_train_partial['identity'].to_list())))
        # import pdb;pdb.set_trace()
    if 0:
        root_dir = '/data/Dataset/PReID/dataset2/'
        # query_dir = root_dir + 'query_a/'
        # gallery_dir = root_dir + 'gallery_a/'
        root_dir = '/data/Dataset/PReID/rep_dataset/'
        query_dir = root_dir + 'rep_f0_query/'
        gallery_dir = root_dir + 'rep_f0_gallery/'

        query_list = glob.glob(query_dir+'*.png')
        gallery_list = glob.glob(gallery_dir+'*.png')

        query_list += gallery_list

        query_idx = list(range(len(query_list)))
        querys = {'filename':query_list,'identity':query_idx}
        df_query = pd.DataFrame(querys)

        df_query_partial = df_query

        def func(fname):
            img = cv2.imread(fname)
            return simple_hist_predictor(img,channel=2,thres=100)
        labels = mmcv.track_parallel_progress(func, df_query_partial['filename'], 8)
        df_query_partial['label'] = labels
        print("==> distribution for hist class")
        print(df_query_partial['label'].value_counts())
    if 0:
        img_dir = '../rep_work_dirs/pseudo_auto_flip_eps055_2/'
        img_list = glob.glob(img_dir+'*/*.png')


        img_idx = list(range(len(img_list)))
        imgs = {'filename':img_list,'identity':img_idx}
        df_img = pd.DataFrame(imgs)

        df_img_partial = df_img

        def func(fname):
            img = cv2.imread(fname)
            return simple_hist_predictor(img,channel=2,thres=100)
        labels = mmcv.track_parallel_progress(func, df_img_partial['filename'], 8)
        df_img_partial['label'] = labels
        print("==> distribution for hist class")
        print(df_img_partial['label'].value_counts())
    if 1:
        root_dir = '/data/Dataset/PReID/dataset2/rep_B/'
        query_dir = root_dir + 'query_b/'
        gallery_dir = root_dir + 'gallery_b/'
        # root_dir = '/data/Dataset/PReID/rep_dataset/'
        # query_dir = root_dir + 'rep_f0_query/'
        # gallery_dir = root_dir + 'rep_f0_gallery/'

        query_list = glob.glob(query_dir+'*.png')
        gallery_list = glob.glob(gallery_dir+'*.png')

        query_list += gallery_list

        query_idx = list(range(len(query_list)))
        querys = {'filename':query_list,'identity':query_idx}
        df_query = pd.DataFrame(querys)

        df_query_partial = df_query

        def func(fname):
            img = cv2.imread(fname)
            return simple_hist_predictor(img,channel=2,thres=100)
        labels = mmcv.track_parallel_progress(func, df_query_partial['filename'], 8)
        df_query_partial['label'] = labels
        print("==> distribution for hist class")
        print(df_query_partial['label'].value_counts())