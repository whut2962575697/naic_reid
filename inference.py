# -*- encoding: utf-8 -*-
'''
@File    :   inference.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 20:18   xin      1.0         None
'''

# -*- encoding: utf-8 -*-
'''
@File    :   inference.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/28 23:01   xin      1.0         None
'''
from dataset.data import read_image
import os
import torch
import numpy as np
import json
from evaluate import eval_func, euclidean_dist, re_rank
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference_val(model,  transform, batch_size, feature_dim, k1=20, k2=6, p=0.3, use_rerank=False):
    q_img_list = os.listdir(r'E:\data\reid\dataset7\query')
    query_list = list()
    qid_list = list()
    qcid_list = list()
    for q_img in q_img_list:
        query_list.append(os.path.join(r'E:\data\reid\dataset7\query', q_img))
        qid_list.append(int(q_img.strip(".png").split("_")[0]))
        qcid_list.append(int(q_img.strip(".png").split("_")[1].strip("c")))

    g_img_list = os.listdir(r'E:\data\reid\dataset7\gallery')
    gallery_list = list()
    gid_list = list()
    gcid_list = list()
    for g_img in g_img_list:
        gallery_list.append(os.path.join(r'E:\data\reid\dataset7\gallery', g_img))
        gid_list.append(int(g_img.strip(".png").split("_")[0]))
        gcid_list.append(int(g_img.strip(".png").split("_")[1].strip("c")))
    img_list = list()
    for q_img in query_list:
        q_img = read_image(q_img)
        q_img = transform(q_img)
        img_list.append(q_img)
    for g_img in gallery_list:
        g_img = read_image(g_img)
        g_img = transform(g_img)
        img_list.append(g_img)
    query_num = len(query_list)
    img_data = torch.Tensor([t.numpy() for t in img_list])

    model = model.to(device)
    model.eval()
    iter_n = len(img_list) // batch_size
    if len(img_list) % batch_size != 0:
        iter_n += 1
    all_feature = list()
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
    gallery_feat = all_feature[query_num:]
    query_feat = all_feature[:query_num]
    if use_rerank:
        distmat = re_rank(query_feat, gallery_feat, k1, k2, p)
    else:
        distmat = euclidean_dist(query_feat, gallery_feat)




    # distmat = euclidean_dist(query_feat, gallery_feat)
    cmc, mAP, _ = eval_func(distmat, np.array(qid_list), np.array(gid_list),
              np.array(qcid_list), np.array(gcid_list))
    print('Validation Result:')
    print(str(k1) + "  -  " + str(k2) + "  -  " + str(p))
    for r in [1, 5, 10]:

        print('CMC Rank-{}: {:.2%}'.format(r, cmc[r - 1]))
    print('mAP: {:.2%}'.format(mAP))
    with open('re_rank.txt', 'a') as f:
        f.write(str(k1)+"  -  "+str(k2)+"  -  "+str(p) + "\n")
        for r in [1, 5, 10]:
            f.write('CMC Rank-{}: {:.2%}'.format(r, cmc[r - 1])+"\n")
        f.write('mAP: {:.2%}'.format(mAP) + "\n")
        f.write('------------------------------------------\n')
        f.write('------------------------------------------\n')
        f.write('\n\n')



def inference_samples(model,  transform, batch_size, feature_dim, k1=20, k2=6, p=0.3, use_rerank=False):
    query_list = list()
    with open(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集/query_a_list.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.split(" ")
            image_name = data[0].split("/")[1]
            img_file = os.path.join(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\query_a', image_name)
            query_list.append(img_file)

    gallery_list = [os.path.join(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\gallery_a', x) for x in
                    os.listdir(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\gallery_a')]
    query_num = len(query_list)
    img_list = list()
    for q_img in query_list:
        q_img = read_image(q_img)
        q_img = transform(q_img)
        img_list.append(q_img)
    for g_img in gallery_list:
        g_img = read_image(g_img)
        g_img = transform(g_img)
        img_list.append(g_img)
    img_data = torch.Tensor([t.numpy() for t in img_list])
    model = model.to(device)
    model.eval()
    iter_n = len(img_list) // batch_size
    if len(img_list) % batch_size != 0:
        iter_n += 1
    all_feature = list()
    for i in range(iter_n):
        print("batch ----%d----" % (i))
        batch_data = img_data[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():

            ff = torch.FloatTensor(batch_data.size(0), feature_dim).zero_()
            for i in range(2):
                if i == 1:
                    batch_data = batch_data.index_select(3, torch.arange(batch_data.size(3) - 1, -1, -1).long())

                outputs= model(batch_data)

                f = outputs.data.cpu()
                ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            all_feature.append(ff)
    all_feature = torch.cat(all_feature)
    gallery_feat = all_feature[query_num:]
    query_feat = all_feature[:query_num]

    if use_rerank:
        print("use re_rank")
        distmat = re_rank(query_feat, gallery_feat, k1, k2, p)
    else:
        distmat = euclidean_dist(query_feat, gallery_feat)
        distmat = distmat.numpy()
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)

    max_200_indices = indices[:, :200]

    res_dict = dict()
    for q_idx in range(num_q):
        print(query_list[q_idx])
        filename = query_list[q_idx][query_list[q_idx].rindex("\\")+1:]
        max_200_files = [gallery_list[i][gallery_list[i].rindex("\\")+1:] for i in max_200_indices[q_idx]]
        res_dict[filename] = max_200_files

    with open(r'submission_A.json', 'w' ,encoding='utf-8') as f:
        json.dump(res_dict, f)


def pseudo_label_samples(model, query_list, gallery_list,  transform, batch_size, k1=20, k2=6, p=0.3):


    query_num = len(query_list)
    img_list = list()
    for q_img in query_list:
        q_img = read_image(q_img)
        q_img = transform(q_img)
        img_list.append(q_img)
    for g_img in gallery_list:
        g_img = read_image(g_img)
        g_img = transform(g_img)
        img_list.append(g_img)
    img_data = torch.Tensor([t.numpy() for t in img_list])
    model = model.to(device)
    model.eval()
    iter_n = len(img_list) // batch_size
    if len(img_list) % batch_size != 0:
        iter_n += 1
    all_feature = list()
    for i in range(iter_n):
        print("batch ----%d----" % (i))
        batch_data = img_data[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            batch_feature = model(batch_data).detach().cpu()
            # ff = torch.FloatTensor(batch_data.size(0), 2048).zero_()
            # for i in range(2):
            #     if i == 1:
            #         batch_data = batch_data.index_select(3, torch.arange(batch_data.size(3) - 1, -1, -1).long())
            #
            #     outputs_1, outputs_2, outputs_3, outputs_4 = model(batch_data)
            #     outputs = torch.cat((outputs_1, outputs_2, outputs_3, outputs_4), 1)
            #     f = outputs.data.cpu()
            #     ff = ff + f
            #
            # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            # ff = ff.div(fnorm.expand_as(ff))
            all_feature.append(batch_feature)
    all_feature = torch.cat(all_feature)
    gallery_feat = all_feature[query_num:]
    query_feat = all_feature[:query_num]

    distmat = re_rank(query_feat, gallery_feat, k1, k2, p)
    distmat = distmat
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    max_200_indices = indices[:, :200]

    res_dict = dict()
    pseudo_res = {"q_imgs": list(), "g_imgs": list(), "probs": list()}
    error_prob = {"q_imgs": list(), "g_imgs": list(), "probs": list()}
    true_prob = {"q_imgs": list(), "g_imgs": list(), "probs": list()}
    for q_idx in range(num_q):
        print(query_list[q_idx])
        filename = query_list[q_idx][query_list[q_idx].rindex("\\")+1:]
        max_200_files = [gallery_list[i][gallery_list[i].rindex("\\")+1:] for i in max_200_indices[q_idx]]
        probs = [distmat[q_idx, i] for i in max_200_indices[q_idx]]

        if max_200_files[0].split("_")[0] != filename.split("_")[0]:
            error_prob["q_imgs"].append(filename)
            error_prob["g_imgs"].append(max_200_files[0])
            error_prob["probs"].append(probs[0])
        for i, prob in enumerate(probs):
            if probs[0]<0.1:
                true_prob["q_imgs"].append(filename)
                true_prob["g_imgs"].append(max_200_files[i])
                true_prob["probs"].append(probs[i])

        for g_filename, prob in zip(max_200_files, probs):
            pseudo_res["q_imgs"].append(filename)
            pseudo_res["g_imgs"].append(g_filename)
            pseudo_res["probs"].append(prob)


        res_dict[filename] = max_200_files

    columns = [u'q_imgs', u'g_imgs', u'probs']
    save_df = pd.DataFrame(pseudo_res,
                           columns=columns)
    save_df.to_csv('pseudo_res.csv')
    save_df = pd.DataFrame(error_prob,
                           columns=columns)
    save_df.to_csv('error_pseudo_res.csv')
    save_df = pd.DataFrame(true_prob,
                           columns=columns)
    save_df.to_csv('true_pseudo_res.csv')



if __name__ == "__main__":
    import torchvision.transforms as T
    from models.baseline import Baseline


    from config import cfg
    from common.sync_bn import convert_model
    from models import build_model

    model = build_model(cfg, 2772)
    para_dict = torch.load(r'E:\data\reid\output/mfn_epoch235.pth')
    model = torch.nn.DataParallel(model)
    model = convert_model(model)
    model.load_state_dict(para_dict)

    transform = T.Compose([
            T.Resize((256, 128)),

            T.ToTensor(),
            # T.Normalize(mean=[0.09661545, 0.18356957, 0.21322473], std=[0.13422933, 0.14724616, 0.19259872])
        ])

    # query_list = list()
    # with open(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集/query_a_list.txt', 'r') as f:
    #     lines = f.readlines()
    #     for i, line in enumerate(lines):
    #         data = line.split(" ")
    #         image_name = data[0].split("/")[1]
    #         img_file = os.path.join(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\query_a', image_name)
    #         query_list.append(img_file)
    #
    # gallery_list = [os.path.join(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\gallery_a', x) for x in
    #                 os.listdir(r'E:\data\reid\初赛A榜测试集\初赛A榜测试集\gallery_a')]
    # query_num = len(query_list)

    # inference_val(model, transform, 64, 2048,  6, 3, 0.8, use_rerank=True)
    # q_img_list = [os.path.join(r'E:\data\reid\dataset5\query', x) for x in os.listdir(r'E:\data\reid\dataset5\query')]
    # g_img_list = [os.path.join(r'E:\data\reid\dataset5\gallery', x) for x in os.listdir(r'E:\data\reid\dataset5\gallery')]
    # pseudo_label_samples(model, query_list, gallery_list, transform, 16, 15, 3, 0.7)
    inference_samples(model, transform, 64, 2048, 8, 3, 0.8, True)
    # tta_inference_samples(model, 64, 15, 3, 0.7)
    # inference_val_with_tta(model, 64)

    # batch_size = 64





    # k1_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # k2_list = [2, 3, 4, 5, 6]
    # p_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for k1 in k1_list:
    #     for k2 in k2_list:
    #         for p in p_list:
    #             inference_val(model, transform, 64, 2048,  k1, k2, p, True)






