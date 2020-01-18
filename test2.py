import cv2
import os
import numpy as np
import json
import time 
import pprint
import pickle
import mmcv
import functools
import pandas as pd
import argparse
import matplotlib
import shutil
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from tqdm import tqdm
from dataset.data import read_image
from evaluate import eval_func, euclidean_dist, re_rank
from dataset import make_dataloader
from utils.swa import specific_bn_update,bn_update
from torch.utils.data import Dataset,DataLoader,SequentialSampler,RandomSampler
from sklearn.cluster import DBSCAN
from scipy import sparse
from rerank_batch import re_ranking_batch,re_ranking_batch_gpu

def simple_hist_predictor(image,channel=2,thres=100): #BGR; by the last channel
    hist = cv2.calcHist([image], [channel], None, [256], [0, 256]) #绘制各个通道的直方图
    if hist[0]>thres:
        return 0
    else:
        return 1
def img_hist_predictor(fname):
    img = cv2.imread(fname)
    return simple_hist_predictor(img,channel=2,thres=100)
device = 'cuda'
# def aqe_func(feat,all_feature,k2,alpha):
#     sims = np.dot(feat.reshape(1,-1),all_feature.T) # 1,N
#     initial_rank = np.argpartition(-sims,range(1,k2+1)) # 1,N
#     weights = sims[0,initial_rank[0,:k2]].reshape((-1,1)) # k2,1
#     weights = np.power(weights,alpha)
#     return np.mean(all_feature[initial_rank[0,:k2],:]*weights,axis=0)
def aqe_func(feat,all_feature,k2,alpha):
    st = time.time()
    sims = np.dot(feat.reshape(1,-1),all_feature.T) # 1,N
    # initial_rank = np.argpartition(-sims,range(1,k2+1)) # 1,N
    initial_rank = np.argpartition(-sims,k2) # 1,N

    weights = sims[0,initial_rank[0,:k2]].reshape((-1,1)) # k2,1
    weights = np.power(weights,alpha)
    return np.mean(all_feature[initial_rank[0,:k2],:]*weights,axis=0)

def aqe_func_gpu(all_feature,k2,alpha,len_slice = 1000):
    all_feature = F.normalize(all_feature, p=2, dim=1)
    gpu_feature = all_feature.cuda()
    T_gpu_feature = gpu_feature.permute(1,0)
    all_feature = all_feature.numpy()

    n_iter = len(all_feature) // len_slice + int(len(all_feature) % len_slice > 0)

    all_features = []

    with tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            # cal sim by gpu
            sims = torch.mm(gpu_feature[i*len_slice:(i+1)*len_slice], T_gpu_feature)
            sims = sims.data.cpu().numpy()
            for sim in sims:
                initial_rank = np.argpartition(-sim,range(1,k2+1)) # 1,N
                # initial_rank = np.argpartition(-sim,k2) # 1,N
                weights = sim[initial_rank[:k2]].reshape((-1,1)) # k2,1
                # weights /= np.max(weights)
                weights = np.power(weights,alpha)
            
                all_features.append(np.mean(all_feature[initial_rank[:k2],:]*weights,axis=0))

            pbar.update(1)

    all_feature = np.stack(all_features,axis=0)

    all_feature = torch.from_numpy(all_feature)
    all_feature = F.normalize(all_feature, p=2, dim=1)
    return all_feature

def predict_pseudo_label(sparse_distmat, eps=0.5, min_points=4, max_points=50,algorithm='brute'):
    dbscaner = DBSCAN(eps = eps, min_samples = min_points,algorithm=algorithm,n_jobs=6,metric='precomputed')
    # dbscaner = DBSCAN(eps = eps, min_samples = min_points,n_jobs=6,metric='precomputed')
    cls_res = dbscaner.fit_predict(sparse_distmat)
    res_dict = dict()
    for i in range(cls_res.shape[0]):
        if cls_res[i] == -1 or cls_res[i] == None:
            continue
        if cls_res[i] not in res_dict.keys():
            res_dict[cls_res[i]] = []

        res_dict[cls_res[i]].append(i)
    filter_res = {}
    for k , v in res_dict.items():
        if len(v) >= min_points and len(v) <= max_points:
            filter_res[k] = v
    # import pdb;pdb.set_trace()
    
    return filter_res

       
def get_sparse_distmat(all_feature,eps,len_slice = 1000,use_gpu=False,dist_k=-1,top_k=35):
    if use_gpu:
        gpu_feature = all_feature.cuda()
    else:
        gpu_feature = all_feature
    n_iter = len(all_feature) // len_slice + int(len(all_feature) % len_slice > 0)
    distmats = []
    kdist = []
    with tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            if use_gpu:
                distmat = euclidean_dist(gpu_feature[i*len_slice:(i+1)*len_slice], gpu_feature).data.cpu().numpy()
            else:
                distmat = euclidean_dist(gpu_feature[i*len_slice:(i+1)*len_slice], gpu_feature).numpy()

            if dist_k>0:
                dist_rank = np.argpartition(distmat,range(1,dist_k+1)) # 1,N
                for j in range(distmat.shape[0]):
                    kdist.append(distmat[j,dist_rank[j,dist_k]])
            if 0:
                initial_rank = np.argpartition(distmat,top_k) # 1,N
                for j in range(distmat.shape[0]):
                    distmat[j,initial_rank[j,top_k:]] = 0
            else:
                distmat[distmat>eps] = 0
            distmats.append(sparse.csr_matrix(distmat))
            
            pbar.update(1)
    if dist_k>0:
        return sparse.vstack(distmats),kdist

    return sparse.vstack(distmats)

def inference_val(args,model,  dataloader,num_query,save_dir, k1=20, k2=6, p=0.3, use_rerank=False,use_flip=False,n_randperm=0,bn_keys=[]):
    model = model.to(device)
    if args.adabn and len(bn_keys)>0:
        print("==> using adabn for specific bn layers")
        specific_bn_update(model,dataloader,cumulative = not args.adabn_emv,bn_keys=bn_keys)
    elif args.adabn:
        print("==> using adabn for all bn layers")
        bn_update(model,dataloader,cumulative = not args.adabn_emv)

    model.eval()
    feats, pids, camids = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            data, pid, camid, _ = batch
            data = data.cuda()

            if use_flip:
                ff = torch.FloatTensor(data.size(0), 2048*2).zero_()
                for i in range(2):
                    # flip
                    if i == 1:
                        data = data.index_select(3, torch.arange(data.size(3) - 1, -1, -1).long().to('cuda'))
                    outputs = model(data)
                    f = outputs.data.cpu()
                    # cat
                    if i == 0:
                        ff[:, :2048] = F.normalize(f, p=2, dim=1)
                    if i == 1:
                        ff[:, 2048:] = F.normalize(f, p=2, dim=1)
                ff = F.normalize(ff, p=2, dim=1)
                # ff = torch.FloatTensor(data.size(0), 2048).zero_()
                # for i in range(2):
                #     if i == 1:
                #         data = data.index_select(3, torch.arange(data.size(3) - 1, -1, -1).long().to('cuda'))
                #     outputs = model(data)
                #     f = outputs.data.cpu()
                #     ff = ff + f
                # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                # ff = ff.div(fnorm.expand_as(ff))
            else:
                ff = model(data).data.cpu()
                ff = F.normalize(ff, p=2, dim=1)

            feats.append(ff)
            pids.append(pid)
            camids.append(camid)
    all_feature = torch.cat(feats, dim=0)
    # all_feature = all_feature[:,:1024+512]
    pids = torch.cat(pids, dim=0)
    camids = torch.cat(camids, dim=0)

    # DBA
    if args.dba:
        k2 = args.dba_k2
        alpha = args.dba_alpha
        assert alpha<0
        print("==>using DBA k2:{} alpha:{}".format(k2,alpha))
        st = time.time()
        
        # [todo] heap sort
        distmat = euclidean_dist(all_feature, all_feature)
        # initial_rank = distmat.numpy().argsort(axis=1)
        initial_rank = np.argpartition(distmat.numpy(),range(1,k2+1)) 

        all_feature = all_feature.numpy()

        V_qe = np.zeros_like(all_feature,dtype=np.float32)
        weights = np.logspace(0,alpha,k2).reshape((-1,1))
        with tqdm(total=len(all_feature)) as pbar:
            for i in range(len(all_feature)):
                V_qe[i,:] = np.mean(all_feature[initial_rank[i,:k2],:]*weights,axis=0)
                pbar.update(1)
        # import pdb;pdb.set_trace()
        all_feature = V_qe
        del V_qe
        all_feature = torch.from_numpy(all_feature)

        fnorm = torch.norm(all_feature, p=2, dim=1, keepdim=True)
        all_feature = all_feature.div(fnorm.expand_as(all_feature))
        print("DBA cost:",time.time()-st)
    # aQE: weight query expansion
    if args.aqe:
        k2 = args.aqe_k2
        alpha = args.aqe_alpha
        
        print("==>using weight query expansion k2: {} alpha: {}".format(k2,alpha))
        st = time.time()
    
        # # [todo] remove norma; normalize is used to to make sure the similiar one is itself
        # all_feature = F.normalize(all_feature, p=2, dim=1)
        # sims = torch.mm(all_feature, all_feature.t()).numpy()

        # # [todo] heap sort
        # # initial_rank = sims.argsort(axis=1)[:,::-1]
        # initial_rank = np.argpartition(-sims,range(1,k2+1))

        # all_feature = all_feature.numpy()

        # V_qe = np.zeros_like(all_feature,dtype=np.float32)

        # # [todo] update query feature only?
        # with tqdm(total=len(all_feature)) as pbar:
        #     for i in range(len(all_feature)):
        #         # get weights from similarity
        #         weights = sims[i,initial_rank[i,:k2]].reshape((-1,1))
        #         # weights = (weights-weights.min())/(weights.max()-weights.min())
        #         weights = np.power(weights,alpha)
        #         # import pdb;pdb.set_trace()
                
        #         V_qe[i,:] = np.mean(all_feature[initial_rank[i,:k2],:]*weights,axis=0)
        #         pbar.update(1)
        # # import pdb;pdb.set_trace()
        # all_feature = V_qe
        # del V_qe
        # all_feature = torch.from_numpy(all_feature)
        # all_feature = F.normalize(all_feature, p=2, dim=1)
        
        # func = functools.partial(aqe_func,all_feature=all_feature,k2=k2,alpha=alpha)
        # all_features = mmcv.track_parallel_progress(func, all_feature, 6)

        # cpu
        # all_feature = F.normalize(all_feature, p=2, dim=1)
        # all_feature = all_feature.numpy()

        # all_features = []
        # with tqdm(total=len(all_feature)) as pbar:
        #     for i in range(len(all_feature)):
        #         all_features.append(aqe_func(all_feature[i],all_feature=all_feature,k2=k2,alpha=alpha))
        #         pbar.update(1)
        # all_feature = np.stack(all_features,axis=0)
        # all_feature = torch.from_numpy(all_feature)
        # all_feature = F.normalize(all_feature, p=2, dim=1)

        all_feature = aqe_func_gpu(all_feature,k2,alpha,len_slice = 2000)
        print("aQE cost:",time.time()-st)
        # import pdb;pdb.set_trace()

    if args.pseudo:
        print("==> using pseudo eps:{} minPoints:{} maxpoints:{}".format(args.pseudo_eps,args.pseudo_minpoints,args.pseudo_maxpoints))
        st = time.time()
        # cal sparse distmat
        all_feature = F.normalize(all_feature, p=2, dim=1)

        # all_distmat = euclidean_dist(all_feature, all_feature).numpy()
        # print(all_distmat[0])
        # pred1 = predict_pseudo_label(all_distmat,args.pseudo_eps,args.pseudo_minpoints,args.pseudo_maxpoints,args.pseudo_algorithm)
        # print(list(pred1.keys())[:10])

        if args.pseudo_visual:
            all_distmat,kdist = get_sparse_distmat(all_feature,eps=args.pseudo_eps+0.1,len_slice=2000,use_gpu=True,dist_k=args.pseudo_minpoints)
            plt.plot(list(range(len(kdist))),np.sort(kdist),linewidth=0.5)
            plt.savefig('eval_kdist.png')
            plt.savefig(save_dir+'eval_kdist.png')            
        else:
            all_distmat = get_sparse_distmat(all_feature,eps=args.pseudo_eps+0.1,len_slice=2000,use_gpu=True)

        # print(all_distmat.todense()[0])

        pseudolabels = predict_pseudo_label(all_distmat,args.pseudo_eps,args.pseudo_minpoints,args.pseudo_maxpoints,args.pseudo_algorithm)
        print("pseudo cost: {}s".format(time.time()-st))
        print("pseudo id cnt:",len(pseudolabels))
        print("pseudo img cnt:",len([x for k,v in pseudolabels.items() for x in v]))
        print("pseudo cost: {}s".format(time.time()-st))
        # print(list(pred.keys())[:10])
    print('feature shape:',all_feature.size())

#
    # for k1 in range(5,10,2):
    #     for k2 in range(2,5,1):
    #         for l in range(5,8):
    #             p = l*0.1

    if n_randperm <=0 :
        k2 = args.k2 
        gallery_feat = all_feature[num_query:]
        query_feat = all_feature[:num_query]
        
        query_pid = pids[:num_query]
        query_camid = camids[:num_query]

        gallery_pid = pids[num_query:]
        gallery_camid = camids[num_query:]
        
        if use_rerank:
            print('==> using rerank')
            # distmat = re_rank(query_feat, gallery_feat, k1, k2, p)
            distmat = re_ranking_batch_gpu(torch.cat([query_feat,gallery_feat],dim=0),num_query,args.k1,args.k2,p)
        else:
            print('==> using euclidean_dist')
            distmat = euclidean_dist(query_feat, gallery_feat)

        cmc, mAP, _ = eval_func(distmat, query_pid.numpy(), gallery_pid.numpy(),query_camid.numpy(), gallery_camid.numpy())
    else:
        k2 = args.k2 
        torch.manual_seed(0)
        cmc = 0
        mAP = 0
        for i in range(n_randperm):
            index = torch.randperm(all_feature.size()[0])
        
            query_feat = all_feature[index][:num_query]
            gallery_feat = all_feature[index][num_query:]

            query_pid = pids[index][:num_query]
            query_camid = camids[index][:num_query]

            gallery_pid = pids[index][num_query:]
            gallery_camid = camids[index][num_query:]

            if use_rerank:
                print('==> using rerank')
                st = time.time()
                # distmat = re_rank(query_feat, gallery_feat, k1, k2, p)
                distmat = re_ranking_batch_gpu(torch.cat([query_feat,gallery_feat],dim=0),num_query,args.k1,args.k2,p)

                print("re_rank cost:",time.time()-st)

            else:
                print('==> using euclidean_dist')
                st = time.time()
                distmat = euclidean_dist(query_feat, gallery_feat)
                print("euclidean_dist cost:",time.time()-st)


            _cmc, _mAP, _ = eval_func(distmat, query_pid.numpy(), gallery_pid.numpy(),query_camid.numpy(), gallery_camid.numpy())
            cmc += _cmc/n_randperm
            mAP += _mAP/n_randperm

    print('Validation Result:')
    if use_rerank:
        print(str(k1) + "  -  " + str(k2) + "  -  " + str(p))
    print('mAP: {:.2%}'.format(mAP))
    for r in [1, 5, 10]:
        print('CMC Rank-{}: {:.2%}'.format(r, cmc[r - 1]))
    print('average of mAP and rank1: {:.2%}'.format((mAP+cmc[0])/2.0))

    with open(save_dir+'eval.txt', 'a') as f:
        if use_rerank:
            f.write('==> using rerank\n')
            f.write(str(k1)+"  -  "+str(k2)+"  -  "+str(p) + "\n")
        else:
            f.write('==> using euclidean_dist\n')

        f.write('mAP: {:.2%}'.format(mAP) + "\n")
        for r in [1, 5, 10]:
            f.write('CMC Rank-{}: {:.2%}'.format(r, cmc[r - 1])+"\n")
        f.write('average of mAP and rank1: {:.2%}\n'.format((mAP+cmc[0])/2.0))

        f.write('------------------------------------------\n')
        f.write('------------------------------------------\n')
        f.write('\n\n')


class ImageDataset(Dataset):
    """RoIs Person ReID Dataset"""

    def __init__(self, img_fnames,transform=None):
        self.img_fnames = img_fnames
        self.transform = transform
    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, index):
        img = self.img_fnames[index]
        img = read_image(img)
        # im = img.resize((self.width,self.height),resample = Image.LANCZOS) # diff1 #1-50ms
        if self.transform is not None:
            img = self.transform(img)
        return img

def get_post_json(distmat, qfnames, gfnames, top_per = 0.7,topk=200):
    res_dict = {}
    for i in range(len(distmat)):
        res_dict[qfnames[i]] = []

    num_q, num_g = distmat.shape
    # get flatten dist ranks
    flatten_dist = distmat.reshape(-1)
    flatten_dist_ranks = np.argsort(flatten_dist)

    gallery_used = np.zeros(len(gfnames))
    
    # got rank1 distmat and split point
    initial_rank = np.argpartition(distmat,range(1,2)) 
    # rank1_dists = distmat[initial_rank[:,0]]
    rank1_dists = []
    for i,r1 in enumerate(initial_rank[:,0]):
        rank1_dists.append(distmat[i,r1])
    rank1_dists = np.array(rank1_dists)
    rank1_dists.sort()
    threshold = rank1_dists[int(len(rank1_dists) * top_per)]  
    print("using threshold:",threshold)
    num_outputs = 0
    num_ignores = 0
    with tqdm(total=len(flatten_dist_ranks)) as pbar:
        for i in range(len(flatten_dist_ranks)):
            q_idx = flatten_dist_ranks[i] // num_g
            g_idx = flatten_dist_ranks[i] % num_g
            #
            if gallery_used[g_idx] == 0:
                if flatten_dist[flatten_dist_ranks[i]] < threshold:
                    gallery_used[g_idx] = 1
                if len(res_dict[qfnames[q_idx]]) < topk:
                    num_outputs += 1
                    res_dict[qfnames[q_idx]].append(gfnames[g_idx])
            else:
                num_ignores += 1
        
            # got full result and early stop
            if num_outputs >= num_q * topk:
                break
            pbar.update(1)
    print("got ignores number:",num_ignores)
    return res_dict
def inference_samples(args,model,  transform, batch_size, query_txt,query_dir,gallery_dir,save_dir,k1=20, k2=6, p=0.3, use_rerank=False,use_flip=False,max_rank=200,bn_keys=[]):
    print("==>load data info..")
    if query_txt != "":
        query_list = list()
        with open(query_txt, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                data = line.split(" ")
                image_name = data[0].split("/")[1]
                img_file = os.path.join(query_dir, image_name)
                query_list.append(img_file)
    else:
        query_list = [os.path.join(query_dir, x) for x in os.listdir(query_dir)]
    gallery_list = [os.path.join(gallery_dir, x) for x in os.listdir(gallery_dir)]
    query_num = len(query_list)
    if args.save_fname != '':
        print(query_list[:10])
        query_list = sorted(query_list)
        print(query_list[:10])
        gallery_list = sorted(gallery_list)
    print("==>build dataloader..")
    image_set = ImageDataset(query_list+gallery_list,transform)
    dataloader = DataLoader(image_set,sampler = SequentialSampler(image_set),batch_size= batch_size,num_workers = 6)
    bn_dataloader = DataLoader(image_set,sampler = RandomSampler(image_set),batch_size= batch_size,num_workers = 6,drop_last = True)
    
    print("==>model inference..")

    model = model.to(device)
    if args.adabn and len(bn_keys)>0:
        print("==> using adabn for specific bn layers")
        specific_bn_update(model,bn_dataloader,cumulative = not args.adabn_emv,bn_keys=bn_keys)
    elif args.adabn:
        print("==> using adabn for all bn layers")
        bn_update(model,bn_dataloader,cumulative = not args.adabn_emv)

    model.eval()
    feats = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            data  = batch
            data = data.cuda()

            if use_flip:
                ff = torch.FloatTensor(data.size(0), 2048*2).zero_()
                for i in range(2):
                    # flip
                    if i == 1:
                        data = data.index_select(3, torch.arange(data.size(3) - 1, -1, -1).long().to('cuda'))
                    outputs = model(data)
                    f = outputs.data.cpu()
                    # cat
                    if i == 0:
                        ff[:, :2048] = F.normalize(f, p=2, dim=1)
                    if i == 1:
                        ff[:, 2048:] = F.normalize(f, p=2, dim=1)
                ff = F.normalize(ff, p=2, dim=1)
            else:
                ff = model(data).data.cpu()
                ff = F.normalize(ff, p=2, dim=1)
            feats.append(ff)

    all_feature = torch.cat(feats, dim=0)

    # DBA
    if args.dba:
        k2 = args.dba_k2
        alpha = args.dba_alpha
        assert alpha<0
        print("==>using DBA k2:{} alpha:{}".format(k2,alpha))
        st = time.time()
        
        # [todo] heap sort
        distmat = euclidean_dist(all_feature, all_feature)
        # initial_rank = distmat.numpy().argsort(axis=1)
        initial_rank = np.argpartition(distmat.numpy(),range(1,k2+1)) 

        all_feature = all_feature.numpy()

        V_qe = np.zeros_like(all_feature,dtype=np.float32)
        weights = np.logspace(0,alpha,k2).reshape((-1,1))
        with tqdm(total=len(all_feature)) as pbar:
            for i in range(len(all_feature)):
                V_qe[i,:] = np.mean(all_feature[initial_rank[i,:k2],:]*weights,axis=0)
                pbar.update(1)
        # import pdb;pdb.set_trace()
        all_feature = V_qe
        del V_qe
        all_feature = torch.from_numpy(all_feature)

        fnorm = torch.norm(all_feature, p=2, dim=1, keepdim=True)
        all_feature = all_feature.div(fnorm.expand_as(all_feature))
        print("DBA cost:",time.time()-st)
    # aQE: weight query expansion
    if args.aqe:
        k2 = args.aqe_k2
        alpha = args.aqe_alpha
        
        print("==>using weight query expansion k2: {} alpha: {}".format(k2,alpha))
        st = time.time()
        
        # fast by gpu
        all_feature = aqe_func_gpu(all_feature,k2,alpha,len_slice = 2000)
        print("aQE cost:",time.time()-st)
    print('feature shape:',all_feature.size())
    if args.pseudo:
        print("==> using pseudo eps:{} minPoints:{} maxpoints:{}".format(args.pseudo_eps,args.pseudo_minpoints,args.pseudo_maxpoints))

        st = time.time()
        
        all_feature = F.normalize(all_feature, p=2, dim=1)
        if args.pseudo_hist:
            print("==> predict histlabel...")
           
            img_filenames = query_list+gallery_list
            img_idx = list(range(len(img_filenames)))
            imgs = {'filename':img_filenames,'identity':img_idx}
            df_img = pd.DataFrame(imgs)
            hist_labels = mmcv.track_parallel_progress(img_hist_predictor, df_img['filename'], 6)
            print("hist label describe..")
            unique_hist_labels = sorted(list(set(hist_labels)))

            hl_idx = []

            hl_query_infos = []
            hl_gallery_infos = []

            for label_idx in range(len(unique_hist_labels)):    
                hl_query_infos.append([])
                hl_gallery_infos.append([])
                hl_idx.append([])
            with tqdm(total = len(img_filenames)) as pbar:
                for idx,info in enumerate(img_filenames):
                    for label_idx in range(len(unique_hist_labels)):    
                        if hist_labels[idx] == unique_hist_labels[label_idx]:
                            if idx<len(query_list):
                                hl_query_infos[label_idx].append(info)
                            else:
                                hl_gallery_infos[label_idx].append(info)
                            hl_idx[label_idx].append(idx)
                    pbar.update(1)
            for label_idx in range(len(unique_hist_labels)):    
                print('hist_label:',unique_hist_labels[label_idx],' query number:',len(hl_query_infos[label_idx]))
                print('hist_label:',unique_hist_labels[label_idx],' gallery number:',len(hl_gallery_infos[label_idx]))
                print('hist_label:',unique_hist_labels[label_idx],' q+g number:',len(hl_query_infos[label_idx])+len(hl_gallery_infos[label_idx]))
                print('hist_label:',unique_hist_labels[label_idx],' idx q+g number:',len(hl_idx[label_idx]))
            # pseudo
            pid = args.pseudo_startid
            camid = 0
            all_list = query_list+gallery_list
            save_path = args.pseudo_savepath
            pseudo_eps = args.pseudo_eps
            pseudo_minpoints = args.pseudo_minpoints
            for label_idx in range(len(unique_hist_labels)):  
                # if label_idx == 0:  
                #     pseudo_eps = 0.6
                # else:
                #     pseudo_eps = 0.75
                if label_idx == 0:  
                    pseudo_eps = 0.65
                else:
                    pseudo_eps = 0.80

            
                feature = all_feature[hl_idx[label_idx]]
                img_list = [all_list[idx] for idx in hl_idx[label_idx]]

                print("==> get sparse distmat!")
                if args.pseudo_visual:
                    all_distmat,kdist = get_sparse_distmat(feature,eps=pseudo_eps+0.05,len_slice=2000,use_gpu=True,dist_k=pseudo_minpoints)
                    plt.plot(list(range(len(kdist))),np.sort(kdist),linewidth=0.5)
                    plt.savefig('test_kdist_hl{}_eps{}_{}.png'.format(label_idx,pseudo_eps,pseudo_minpoints))
                    plt.savefig(save_dir+'test_kdist_hl{}_eps{}_{}.png'.format(label_idx,pseudo_eps,pseudo_minpoints))
                else:
                    all_distmat = get_sparse_distmat(feature,eps=pseudo_eps+0.05,len_slice=2000,use_gpu=True)

                print("==> predict pseudo label!")
                pseudolabels = predict_pseudo_label(all_distmat,pseudo_eps,pseudo_minpoints,args.pseudo_maxpoints,args.pseudo_algorithm)
                print("==> using pseudo eps:{} minPoints:{} maxpoints:{}".format(pseudo_eps,pseudo_minpoints,args.pseudo_maxpoints))
                print("pseudo cost: {}s".format(time.time()-st))
                print("pseudo id cnt:",len(pseudolabels))
                print("pseudo img cnt:",len([x for k,v in pseudolabels.items() for x in v]))

                if label_idx == 0:
                    sf = 1
                else:
                    sf = 1
                sample_id_cnt = 0
                sample_file_cnt = 0
                nignore_query = 0
                for i,(k,v) in enumerate(pseudolabels.items()):
                    if i%sf !=0:
                        continue
                    # query_cnt = 0
                    # for _index in pseudolabels[k]:
                    #     if _index<len(query_list):
                    #         query_cnt += 1
                    # if query_cnt>=2:
                    #     nignore_query += 1
                    #     continue
                    os.makedirs(os.path.join(save_path, str(pid)),exist_ok=True)

                    for _index in pseudolabels[k]:
                        filename = img_list[_index].split("/")[-1]
                        new_filename = str(pid)+"_c"+str(camid)+".png" 
                        shutil.copy(img_list[_index], os.path.join(save_path, str(pid), new_filename))
                        camid += 1
                        sample_file_cnt += 1
                    sample_id_cnt += 1
                    pid += 1
                print("pseudo ignore id cnt:",nignore_query)
                print("sample id cnt:",sample_id_cnt)
                print("sample file cnt:",sample_file_cnt)                
        else:
            if args.pseudo_visual:
                all_distmat,kdist = get_sparse_distmat(all_feature,eps=args.pseudo_eps+0.05,len_slice=2000,use_gpu=True,dist_k=args.pseudo_minpoints)
                plt.plot(list(range(len(kdist))),np.sort(kdist),linewidth=0.5)
                plt.savefig('test_kdist.png')
                plt.savefig(save_dir+'test_kdist.png')            
            else:
                all_distmat = get_sparse_distmat(all_feature,eps=args.pseudo_eps+0.05,len_slice=2000,use_gpu=True)

            # print(all_distmat.todense()[0])

            pseudolabels = predict_pseudo_label(all_distmat,args.pseudo_eps,args.pseudo_minpoints,args.pseudo_maxpoints,args.pseudo_algorithm)
            print("pseudo cost: {}s".format(time.time()-st))
            print("pseudo id cnt:",len(pseudolabels))
            print("pseudo img cnt:",len([x for k,v in pseudolabels.items() for x in v]))

            # # save
            all_list = query_list+gallery_list
            save_path = args.pseudo_savepath

            pid = args.pseudo_startid
            camid = 0
            nignore_query = 0
            for k,v in pseudolabels.items():
                os.makedirs(os.path.join(save_path, str(pid)),exist_ok=True)
                # [fileter]
                query_cnt = 0
                for _index in pseudolabels[k]:
                    if _index<len(query_list):
                        query_cnt += 1
                if query_cnt>=4:
                    nignore_query += 1
                    continue
                for _index in pseudolabels[k]:
                    filename = all_list[_index].split("/")[-1]
                    new_filename = str(pid)+"_c"+str(camid)+".png" 
                    shutil.copy(all_list[_index], os.path.join(save_path, str(pid), new_filename))
                    camid += 1
                pid += 1
            print("pseudo ignore id cnt:",nignore_query)
    else:
        gallery_feat = all_feature[query_num:]
        query_feat = all_feature[:query_num]

        if use_rerank:
            print("==>use re_rank")
            st = time.time()
            k2 = args.k2 
            # distmat = re_rank(query_feat, gallery_feat, k1, k2, p)
            num_query = len(query_feat)
            print("using k1:{} k2:{} lambda:{}".format(args.k1,args.k2,p))
            distmat = re_ranking_batch_gpu(torch.cat([query_feat,gallery_feat],dim=0),num_query,args.k1,args.k2,p)

            print("re_rank cost:",time.time()-st)

        else:
            print("==>use euclidean_dist")
            st = time.time()
            distmat = euclidean_dist(query_feat, gallery_feat)
            print("euclidean_dist cost:",time.time()-st)

            distmat = distmat.numpy()

        num_q, num_g = distmat.shape
        print("==>saving..")
        if args.post:
            qfnames = [fname.split('/')[-1] for fname in query_list]
            gfnames = [fname.split('/')[-1] for fname in gallery_list]
            st = time.time()
            print("post json using top_per:",args.post_top_per)
            res_dict = get_post_json(distmat,qfnames,gfnames,args.post_top_per)
            print("post cost:",time.time()-st)
        else:
            # [todo] fast test
            print("==>sorting..")
            st = time.time()
            indices = np.argsort(distmat, axis=1)
            print("argsort cost:",time.time()-st)
            # print(indices[:2, :max_rank])
            # st = time.time()
            # indices = np.argpartition( distmat, range(1,max_rank+1))
            # print("argpartition cost:",time.time()-st)
            # print(indices[:2, :max_rank])

            max_200_indices = indices[:, :max_rank]
            res_dict = dict()
            for q_idx in range(num_q):
                filename = query_list[q_idx].split('/')[-1]
                max_200_files = [gallery_list[i].split('/')[-1] for i in max_200_indices[q_idx]]
                res_dict[filename] = max_200_files
        if args.dba:
            save_fname = 'sub_dba.json'
        elif args.aqe:
            save_fname = 'sub_aqe.json'
        else:
            save_fname = 'sub.json'
        if use_rerank:
            save_fname = 'rerank_'+save_fname
        if args.adabn:
            if args.adabn_all:
                save_fname = 'adabnall_'+save_fname
            else:
                save_fname = 'adabn_'+save_fname
        if use_flip:
            save_fname = 'flip_'+save_fname
        if args.post:
            save_fname = 'post_'+save_fname
        save_fname = args.save_fname+save_fname
        print('savefname:',save_fname)
        with open(save_dir+save_fname, 'w' ,encoding='utf-8') as f:
            json.dump(res_dict, f)
        with open(save_dir+save_fname.replace('.json','.pkl'),'wb') as fid:
            pickle.dump(distmat,fid, -1)

if __name__ == "__main__":
    import torchvision.transforms as T
    from models.baseline import Baseline


    from config import cfg
    from common.sync_bn import convert_model
    from models import build_model

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument('--rerank',action='store_true',help='whether to rerank')
    parser.add_argument('--sub',action='store_true',help='whether to sub')

    parser.add_argument('--flip',action='store_true',help='whether to flip test')


    parser.add_argument("--k1", default=8, help="", type=int)
    parser.add_argument("--k2", default=3, help="", type=int)
    parser.add_argument("--lambda_value", default=0.8, help="", type=float)
    parser.add_argument("--max_rank", default=200, help="", type=int)

    parser.add_argument('--dba',action='store_true',help='whether to dba')
    parser.add_argument("--dba_k2", default=10, help="", type=int)

    parser.add_argument("--dba_alpha", default=-3.0, help="", type=float)

    parser.add_argument('--aqe',action='store_true',help='whether to aqe')
    parser.add_argument("--aqe_k2", default=5, help="", type=int)
    parser.add_argument("--aqe_alpha", default=3.0, help="", type=float)
    
    parser.add_argument("--query_txt", default="", help="path to query file", type=str)
    parser.add_argument("--query_dir", default="", help="path to query file", type=str)
    parser.add_argument("--gallery_dir", default="", help="path to query file", type=str)

    parser.add_argument('--adabn',action='store_true',help='whether to adabn')
    parser.add_argument("--adabn_emv",action='store_true',help='whether to adabn by exponential moving average')
    parser.add_argument("--adabn_all",action='store_true',help='whether to adabn for all layers')

    parser.add_argument('--pseudo',action='store_true',help='whether to pseudo')
    parser.add_argument('--pseudo_hist',action='store_true',help='whether to pseudo')
    parser.add_argument("--pseudo_eps",default=0.5, help="", type=float)
    parser.add_argument("--pseudo_minpoints",default=2, help="", type=int)
    parser.add_argument("--pseudo_maxpoints",default=50, help="", type=int)
    parser.add_argument("--pseudo_algorithm",default='brute', help="", type=str)
    parser.add_argument('--pseudo_visual',action='store_true',help='whether to pseudo')
    parser.add_argument('--pseudo_savepath',default='../rep_work_dirs/pseudo', help="", type=str)
    parser.add_argument("--pseudo_startid",default=20000, help="", type=int)



    parser.add_argument('--post',action='store_true',help='whether to post')
    parser.add_argument("--post_top_per",default=0.7, help="", type=float)

    parser.add_argument("--save_fname",default='', help="", type=str)


    parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # print(cfg)
    dict_args = {}
    dict_args.update(vars(args))
    print(pprint.pformat(dict_args))

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_gpus = torch.cuda.device_count()
    train_dl, val_dl, num_query, num_classes = make_dataloader(cfg, num_gpus)
    
    print("==> build model..")
    model = build_model(cfg, num_classes)
    print(model)
    print("==> load params..")
    param_dict = torch.load(cfg.TEST.WEIGHT)
    model = torch.nn.DataParallel(model)
    if cfg.SOLVER.SYNCBN:
        print("convert_model to syncbn")
        model = convert_model(model)
    # 
    param_dict = {k.replace('module.',''): v for k, v in param_dict.items() }
    print('unloaded_param:')
    print([k for k, v in model.state_dict().items() if k.replace('module.','') not in param_dict or param_dict[k.replace('module.','')].size() != v.size()])
    for i in model.state_dict():
        model.state_dict()[i].copy_(param_dict[i.replace('module.','')])
    # model.load_state_dict(param_dict)

    transform = val_dl.dataset.transform
    print(transform)
    
    # normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    # transform = [
    #     T.Resize(cfg.INPUT.SIZE_TEST),
    #     T.ColorJitter(brightness=cfg.INPUT.CJ_BRIGHNESS, contrast=cfg.INPUT.CJ_CONTRAST,saturation=cfg.INPUT.CJ_SATURATION, hue=cfg.INPUT.CJ_HUE),
    #     T.ToTensor(),
    #     # normalize_transform
    # ]
    # if cfg.INPUT.NORMALIZATION:
    #     transform.append(normalize_transform)

    # transform = T.Compose(transform)
    # print(transform)

    query_dir = args.query_dir
    gallery_dir = args.gallery_dir

    bn_keys = []
    if not args.adabn_all:
        if cfg.MODEL.NAME in ['baseline','cosine_baseline']:
            bn_keys += ['bottleneck']
        if cfg.MODEL.NAME == 'mfn':
            # bn_keys += ['classifier','classifier1','classifier2','classifier3']
            bn_keys += ['classifier.add_block1.0','classifier.add_block.0',\
                        'classifier1.add_block1.0','classifier1.add_block.0',\
                        'classifier2.add_block.0','classifier2.add_block1.1',\
                    'classifier3.add_block.0','classifier3.add_block1.1']
        if cfg.MODEL.NAME == "mgn":
            bn_keys += ['reduction_0.1','reduction_1.1','reduction_2.1','reduction_3.1',\
                        'reduction_4.1','reduction_5.1','reduction_6.1','reduction_7.1']
    
    if args.sub == False:
        # num_query = len(val_dl.dataset)//13
        print("num_query:",num_query)
        inference_val(args,model,val_dl,num_query,cfg.OUTPUT_DIR, args.k1,args.k2, args.lambda_value, \
                        use_rerank=args.rerank,use_flip=args.flip,n_randperm=cfg.TEST.RANDOMPERM,\
                        bn_keys = bn_keys)
    else:
        inference_samples(args,model, transform, 256,args.query_txt, query_dir,gallery_dir,cfg.OUTPUT_DIR, args.k1,args.k2, args.lambda_value,\
                        use_rerank=args.rerank,use_flip=args.flip,max_rank=args.max_rank,\
                        bn_keys = bn_keys)