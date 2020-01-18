#!/usr/bin/env python2/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22.
- This version accepts distance matrix instead of raw features.
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""


import numpy as np
from evaluate import eval_func, euclidean_dist
from scipy import sparse
import torch
import time
from tqdm import tqdm
# from memory_profiler import profile


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    # initial_rank = np.argsort(original_dist).astype(np.int32)
    # # fast sort top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1)).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)

        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist


def calculate_V(initial_rank, all_feature_len, dis_i_qg, i,  k1):
    # dis_i_qg = euclidean_dist(torch.tensor([all_feature[i].numpy()]), all_feature).numpy()

    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    # print(forward_k_neigh_index)
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]

    fi = np.where(backward_k_neigh_index == i)[0]
    k_reciprocal_index = forward_k_neigh_index[fi]
    k_reciprocal_expansion_index = k_reciprocal_index
    for j in range(len(k_reciprocal_index)):
        candidate = k_reciprocal_index[j]
        candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2.)) + 1]
        candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                           :int(np.around(k1 / 2.)) + 1]
        fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
        candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
        if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                candidate_k_reciprocal_index):
            k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

    k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
    # print(k_reciprocal_expansion_index)
    weight = np.exp(-dis_i_qg[k_reciprocal_expansion_index])
    # print(weight)
    V = np.zeros(( all_feature_len)).astype(np.float32)
    V[k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    return V, k_reciprocal_expansion_index, weight



def re_ranking_batch(all_feature, q_num, k1, k2, lambda_value, len_slice=1000):

    # calculate (q+g)*(q+g)
    initial_rank = np.zeros((len(all_feature), k1+1)).astype(np.int32)

    original_dist = np.zeros((q_num, len(all_feature)))

    s_time = time.time()

    n_iter = len(all_feature) // len_slice + int(len(all_feature) % len_slice > 0)

    with tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            dis_i_qg = euclidean_dist(all_feature[i*len_slice:(i+1)*len_slice], all_feature).data.cpu().numpy()
            initial_i_rank = np.argpartition(dis_i_qg, range(1, k1 + 1), ).astype(np.int32)[:, :k1 + 1]
            initial_rank[i*len_slice:(i+1)*len_slice] = initial_i_rank
            pbar.update(1)
    # print(initial_rank[0])

    end_time = time.time()
    print("rank time : %s" % (end_time-s_time))

    all_V = []

    s_time = time.time()

    n_iter = len(all_feature) // len_slice + int(len(all_feature) % len_slice > 0)


    with tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            dis_i_qg = euclidean_dist(all_feature[i * len_slice:(i + 1) * len_slice], all_feature).data.cpu().numpy()
            for ks in range(dis_i_qg.shape[0]):
                r_k = i*len_slice+ks
                dis_i_qg[ks] = np.power(dis_i_qg[ks], 2).astype(np.float32)
                dis_i_qg[ks] = 1. * dis_i_qg[ks] / np.max(dis_i_qg[ks])
                if r_k < q_num:
                    original_dist[r_k] = dis_i_qg[ks]
                V ,k_reciprocal_expansion_index, weight = calculate_V(initial_rank, len(all_feature), dis_i_qg[ks], r_k, k1)
                # if r_k == 0:
                #     print(k_reciprocal_expansion_index)
                #     print(weight)
                #     print(dis_i_qg[ks])
                all_V.append(sparse.csr_matrix(V))

            pbar.update(1)

    all_V = sparse.vstack(all_V)
    # print(all_V.getrow(0).toarray())
    end_time = time.time()
    print("calculate V time : %s" % (end_time - s_time))
    # print(all_V.todense()[0])

    all_V_qe = []
    s_time = time.time()
    for i in range(len(all_feature)):
        temp_V = np.zeros((k2, len(all_feature)))
        for l, row_index in enumerate(initial_rank[i, :k2]):
            temp_V[l, :] = all_V.getrow(row_index).toarray()[0]


        V_qe = np.mean(temp_V, axis=0)
        all_V_qe.append(sparse.csr_matrix(V_qe))
    all_V_qe = sparse.vstack(all_V_qe)
    # print(all_V_qe.todense()[0])
    del all_V
    end_time = time.time()
    print("calculate V_qe time : %s" % (end_time - s_time))

    invIndex = []
    for i in range(len(all_feature)):
        invIndex.append(np.where(all_V_qe.getcol(i).toarray().transpose()[0] != 0)[0])
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(q_num):
        temp_min = np.zeros(shape=[1, len(all_feature)], dtype=np.float32)

        indNonZero = np.where(all_V_qe.getrow(i).toarray()[0] != 0)[0]

        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        # print(indImages)
        for j in range(len(indNonZero)):
            # print(indNonZero[j])
            c = all_V_qe.getrow(i).getcol(indNonZero[j]).toarray()[0, 0]
            # print(c)
            # print(indImages[j])

            t_min = np.zeros((indImages[j].shape[0]))
            for kk in range(indImages[j].shape[0]):
                temp_d = all_V_qe.getrow(indImages[j][kk]).getcol(indNonZero[j]).toarray()[0, 0]
                t_min[kk] = np.minimum(c, temp_d)
            # print(t_min)

            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + t_min
            # temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
            #                                                                    V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)
    # print(jaccard_dist[0])
    # print(original_dist[0])
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del all_V_qe
    del jaccard_dist
    final_dist = final_dist[:q_num, q_num:]
    return final_dist

def re_ranking_batch_gpu(all_feature, q_num, k1, k2, lambda_value, len_slice=1000):

    # calculate (q+g)*(q+g)
    initial_rank = np.zeros((len(all_feature), k1+1)).astype(np.int32)

    original_dist = np.zeros((q_num, len(all_feature)))
    gpu_features = all_feature.cuda()
    s_time = time.time()

    n_iter = len(all_feature) // len_slice + int(len(all_feature) % len_slice > 0)

    with tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            dis_i_qg = euclidean_dist(gpu_features[i*len_slice:(i+1)*len_slice], gpu_features).data.cpu().numpy()
            initial_i_rank = np.argpartition(dis_i_qg, range(1, k1 + 1), ).astype(np.int32)[:, :k1 + 1]
            initial_rank[i*len_slice:(i+1)*len_slice] = initial_i_rank
            pbar.update(1)
    # print(initial_rank[0])

    end_time = time.time()
    print("rank time : %s" % (end_time-s_time))

    all_V = []

    s_time = time.time()

    n_iter = len(all_feature) // len_slice + int(len(all_feature) % len_slice > 0)


    with tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            dis_i_qg = euclidean_dist(gpu_features[i * len_slice:(i + 1) * len_slice], gpu_features).data.cpu().numpy()
            for ks in range(dis_i_qg.shape[0]):
                r_k = i*len_slice+ks
                dis_i_qg[ks] = np.power(dis_i_qg[ks], 2).astype(np.float32)
                dis_i_qg[ks] = 1. * dis_i_qg[ks] / np.max(dis_i_qg[ks])
                if r_k < q_num:
                    original_dist[r_k] = dis_i_qg[ks]
                V ,k_reciprocal_expansion_index, weight = calculate_V(initial_rank, len(all_feature), dis_i_qg[ks], r_k, k1)
                # if r_k == 0:
                #     print(k_reciprocal_expansion_index)
                #     print(weight)
                #     print(dis_i_qg[ks])
                all_V.append(sparse.csr_matrix(V))

            pbar.update(1)

    all_V = sparse.vstack(all_V)
    # print(all_V.getrow(0).toarray())
    end_time = time.time()
    print("calculate V time : %s" % (end_time - s_time))
    # print(all_V.todense()[0])

    all_V_qe = []
    s_time = time.time()
    for i in range(len(all_feature)):
        temp_V = np.zeros((k2, len(all_feature)))
        for l, row_index in enumerate(initial_rank[i, :k2]):
            temp_V[l, :] = all_V.getrow(row_index).toarray()[0]


        V_qe = np.mean(temp_V, axis=0)
        all_V_qe.append(sparse.csr_matrix(V_qe))
    all_V_qe = sparse.vstack(all_V_qe)
    # print(all_V_qe.todense()[0])
    del all_V
    end_time = time.time()
    print("calculate V_qe time : %s" % (end_time - s_time))

    invIndex = []
    for i in range(len(all_feature)):
        invIndex.append(np.where(all_V_qe.getcol(i).toarray().transpose()[0] != 0)[0])
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    with tqdm(total=q_num) as pbar:
        for i in range(q_num):
            temp_min = np.zeros(shape=[1, len(all_feature)], dtype=np.float32)

            indNonZero = np.where(all_V_qe.getrow(i).toarray()[0] != 0)[0]

            indImages = []
            indImages = [invIndex[ind] for ind in indNonZero]
            # print(indImages)
            for j in range(len(indNonZero)):
                # print(indNonZero[j])
                c = all_V_qe.getrow(i).getcol(indNonZero[j]).toarray()[0, 0]
                # print(c)
                # print(indImages[j])

                t_min = np.zeros((indImages[j].shape[0]))
                for kk in range(indImages[j].shape[0]):
                    temp_d = all_V_qe.getrow(indImages[j][kk]).getcol(indNonZero[j]).toarray()[0, 0]
                    t_min[kk] = np.minimum(c, temp_d)
                # print(t_min)

                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + t_min
                # temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                #                                                                    V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2. - temp_min)
            pbar.update(1)
    # print(jaccard_dist[0])
    # print(original_dist[0])
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del all_V_qe
    del jaccard_dist
    final_dist = final_dist[:q_num, q_num:]
    return final_dist


# def re_ranking_batch(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
#
#     # The following naming, e.g. gallery_num, is different from outer scope.
#     # Don't care about it.
#
#     original_dist = np.concatenate(
#       [np.concatenate([q_q_dist, q_g_dist], axis=1),
#        np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
#       axis=0)
#     original_dist = np.power(original_dist, 2).astype(np.float32)
#     original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
#     V = np.zeros_like(original_dist).astype(np.float32)
#     # initial_rank = np.argsort(original_dist).astype(np.int32)
#     # # fast sort top K1+1
#     initial_rank = np.argpartition( original_dist, range(1,k1+1)).astype(np.int32)
#
#     query_num = q_g_dist.shape[0]
#     gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
#     all_num = gallery_num
#
#     for i in range(all_num):
#         # k-reciprocal neighbors
#         forward_k_neigh_index = initial_rank[i,:k1+1]
#         backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
#         fi = np.where(backward_k_neigh_index==i)[0]
#         k_reciprocal_index = forward_k_neigh_index[fi]
#         k_reciprocal_expansion_index = k_reciprocal_index
#         for j in range(len(k_reciprocal_index)):
#             candidate = k_reciprocal_index[j]
#             candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
#             candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
#             fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
#             candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
#             if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
#                 k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
#
#         k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
#         weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
#         V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
#     original_dist = original_dist[:query_num,]
#     if k2 != 1:
#         V_qe = np.zeros_like(V,dtype=np.float32)
#         for i in range(all_num):
#             V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
#
#         # # DBA
#         # alpha = -3.0
#         # weights = np.logspace(0,alpha,k2).reshape((-1,1))
#         # for i in range(all_num):
#         #     V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:]*weights,axis=0)
#
#         V = V_qe
#         del V_qe
#     del initial_rank
#     invIndex = []
#     for i in range(gallery_num):
#         invIndex.append(np.where(V[:,i] != 0)[0])
#
#     jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)
#
#
#     for i in range(query_num):
#         temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
#         indNonZero = np.where(V[i,:] != 0)[0]
#         indImages = []
#         indImages = [invIndex[ind] for ind in indNonZero]
#         for j in range(len(indNonZero)):
#             temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
#         jaccard_dist[i] = 1-temp_min/(2.-temp_min)
#
#     final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
#     del original_dist
#     del V
#     del jaccard_dist
#     final_dist = final_dist[:query_num,query_num:]
#     return final_dist