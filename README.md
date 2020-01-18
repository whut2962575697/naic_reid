# naic_reid
# This is Top19's Code for the Person ReID Compitition of NAIC(首届“全国人工智能大赛”（行人重识别 Person ReID 赛项）)

### Dependencies
* python==3.6
* pandas==0.25.1
* hickle==3.4.5
* tqdm==4.36.1
* opencv_python==4.1.1.26
* scikit_image==0.15.0
* mlconfig==0.0.4
* visdom==0.1.8.9
* torch==1.0.0
* torchvision==0.2.0
* yacs==0.1.6
* numpy==1.17.4
* scipy==1.3.1
* apex==0.9.10dev
* ipython==7.10.0
* Pillow==6.2.1
* skimage==0.0

# Prerequisites
you can find the compition at https://www.kesci.com/home/competition/5d90401cd8fc4f002da8e7be/content/2

In order to run this project you will need:

- Python3 (tested with Python 3.6.0)
- PyTorch deep learning framework (tested with version 1.0.0)
- rep dataset contains almost 86k train images and 71k test images for test A, 166k for test B, so the training phase requires ~24GB GPU memory. If your GPU doesn't have sufficient memory, please reduce batch size or reduce image train-size.

# Team members
- [whxin](https://github.com/whut2962575697 "whxin") 
- [John-Yao](https://github.com/John-Yao "John-Yao")
- [gogozhu](https://github.com/gogozhu "gogozhu")
- [dbofseuofhust](https://github.com/dbofseuofhust "dbofseuofhust")
- [douzi](https://github.com/douzi0248/Re-ID   "douzi") 

# Support
- [x] Multi-GPU and SyncBN
- [x] fp16

# Models
- [x] Strong Baseline
- [x] MGN
- [x] MFN(Comming Soon)

# Tricks
- [x] DataAugmention(RandomErase + ColorJittering + RandomHorizontallyFlip + Padding + RandomCrop)
- [x] WarmUp + MultiStepLR 
- [x] PxK = 16x6 (Batch Size = 96)
- [x] ArcFace
- [x] BackBone resnet101_ibn_a
- [x] Size 384*192
- [x] TripletLoss+SoftmaxLoss
- [x] AQE(GPU only cost 20s for test a)
- [x] adabn
- [x] gem
- [x] all data training(not include single pid image)
- [x] Batch GPU ReRanking
- [x] Pseudo Label (dbscan) + Ensemble
- [x] Multi Triplet-Margine Ensemble

1. Due to the characteristics of the dataset, we find Color Jittering can greatly improve model performance.And at the same time we find that using Color Jittering with a 0.5 probability is better.
2. AQE(average query expansion) can replace rerank as a post-processing method, which can imporve evaluation indicators by 1%~2%.There are two aqe methods in our codes, one is based on cpu and the other is based on gpu, which can speed up about 10 times.About AQE,please follow this paper [Fine-tuning CNN Image Retrieval with No Human Annotation](https://github.com/filipradenovic/cnnimageretrieval-pytorch "Fine-tuning CNN Image Retrieval with No Human Annotation")  
3. Adabn can improve mfn performance and baseline with arccos, but there is no improvement in baeline with arcface.
4. Batch GPU ReRanking is modified from the original rerank, using the GPU to calculate the distance, and using sparse matrix storage, which can save memory and time.But due to the limitation of test B submission, this method was not adopted in the end.
5. Pseudo Label is a trick commonly used in deep learning competitions.We use a trained model to cluster the test set, and the clustering method we choose is dbscan which has two hyperparameters to control the clustering effect.
6. Multi Triplet-Margine Ensemble means ensemble two models with different triplet loss margin (we choose 0.5 and 0.3). Due to time constraints, we gave up the trick in the test B.




## Test A 
| model | size | backbone | trick | performance |
| :-----| ----: | :----: | :----: | :----: |
| mgn | 384x128 | resnet50-ibn | adabn + gem | 0.77808439 |
| mgn | 384x144 | resnet50-ibn | adabn + gem | 0.78023715 |
| mgn | 384x144 | resnet50-ibn | adabn + gem + aqe | 0.78967123 |
| mgn | 384x144 | resnet50-ibn | gem + aqe + cj | 0.79911998 |
| baseline | 384x192 | resnet50-ibn | adabn + gem + cosface | 0.76778309 |
| baseline | 384x192 | resnet50-ibn | gem + cosface + cj | 0.79208766 |
| baseline | 384x192 | resnet50-ibn | avg + cosface + cj | 0.79478573 |
| baseline | 384x192 | resnet101-ibn | avg + cosface + cj | 0.80346376 |
| baseline | 384x192 | resnet101-ibn | avg + cosface + cj + all_data | 0.80577292 |
| baseline | 384x192 | resnet101-ibn | avg + cosface + cj05 + all_data | 0.80686313 |
| baseline | 384x192 | resnet101-ibn | avg + arcfaceface + cj05 + all_data | 0.819 |


## Test B public
| model | size | backbone | trick | performance |
| :-----| ----: | :----: | :----: | :----: |
| baseline | 384x192 | resnet101-ibn | (avg + arcfaceface + cj05 + all_data) + (avg + arcfaceface + cj05 + all_data + pseudo label) ensemble | 0.81488409 |

## Test B private 
0.80066173




# Implementation
## prepare_data
you can put the data in rep_dir, and get the result data in save_dir
### split 0.85 train data
1. train minist2 data (0.85 train data (each pid >=2))
2. train data (0.85 train data)
3. train minist4 data (0.85 train data (each pid >=4))
### all train data
1. trainVal2 data (all train data (each pid >= 2))
2. trainVal data (all train data)
### step 1 modify prepare_rep2.py
```python
    root_dir = '/data/Dataset/PReID/'  # dataset root
    rep_dir = root_dir+'dataset2/'  # rep dataset

    save_dir = root_dir+'rep_dataset/' # save path
```
### step 2 run prepare_rep2.py

## Train
for baseline model, you can choose arccos baseline or arcface baseline by modifing 'MODEL.BASELINE.COSINE_LOSS_TYPE'

### setp1 modify train sh file

```
 PRETRAIN=resnet101_ibn_a.pth.tar
 DATA_DIR='your data dir'
 SAVE_DIR='your save dir' #(h, w)
 TRAIN_PATH='your train folder'
 QUERY_PATH='your query folder'
 GALLERY_PATH='your gallery folder'
 CUDA_VISIBLE_DEVICES=5 python train.py --config_file='configs/naic/arcface_baseline.yml' \
     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "10" SOLVER.STEPS "[40, 70]" SOLVER.MAX_EPOCHS "90" SOLVER.START_SAVE_EPOCH "75" SOLVER.EVAL_PERIOD "2" \
     SOLVER.IMS_PER_BATCH "96" DATALOADER.NUM_INSTANCE "6" \
     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "('${TRAIN_PATH}')" DATASETS.QUERY_PATH "('${QUERY_PATH}')"        DATASETS.GALLERY_PATH "('${GALLERY_PATH}')"\
     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
     OUTPUT_DIR "('${SAVE_DIR}')" 

```

### step 2 run ./shells/rep_train_bl.sh


## Test
### step1 modify test sh file

```
DATA_DIR= 'your data dir'
PRETRAIN=resnet101_ibn_a.pth.tar
MODEL_DIR=your model dir #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
SAVE_DIR=${MODEL_DIR}eval/
 
   --flip \
   --aqe --aqe_k2 7 --aqe_alpha 3.0 \
 CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
     TEST.IMS_PER_BATCH "128" TEST.RANDOMPERM "5" \
     MODEL.BASELINE.TPL_WEIGHT "1.0" MODEL.BASELINE.CE_WEIGHT "0.33" MODEL.LABEL_SMOOTH "False" MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "your train data folder"\
     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
     OUTPUT_DIR "('${SAVE_DIR}')" \
     TEST.WEIGHT "${WEIGHT}"
```

### step 2 run ./shells/rep_test_bl.sh

## pseudo label
### step1 modify test sh file for pseudo label
```
# [20191231] testb pseudo

QUERY_DIR=${ROOT_DIR}dataset2/rep_B/query_b/
GALLERY_DIR=${ROOT_DIR}dataset2/rep_B/gallery_b/
DATA_DIR=${ROOT_DIR}rep_dataset/
PRETRAIN=../weights/resnet101_ibn_a.pth.tar
MODEL_DIR=your model dir #(h, w)
WEIGHT=${MODEL_DIR}cosine_baseline_epoch90.pth
SAVE_DIR=${MODEL_DIR}eval/


 SAVE_DIR=${MODEL_DIR}sub/
 CUDA_VISIBLE_DEVICES=2 python test2.py --config_file='configs/naic/arcface_baseline.yml' \
     --sub \
     --pseudo --pseudo_hist --pseudo_visual --pseudo_algorithm "auto" --pseudo_eps 0.55 --pseudo_minpoints 3 --pseudo_maxpoints 100 --pseudo_savepath '../rep_work_dirs/testb_pseudo_hist_065_080/'\
     --query_dir ${QUERY_DIR}\
     --gallery_dir ${GALLERY_DIR}\
     MODEL.BASELINE.S "30.0" MODEL.BASELINE.M "0.35" \
     MODEL.BASELINE.COSINE_LOSS_TYPE 'ArcCos' \
     INPUT.SIZE_TRAIN "([384,192])" INPUT.SIZE_TEST "([384,192])" \
     MODEL.NAME "cosine_baseline" MODEL.BACKBONE "('resnet101_ibn_a')" MODEL.BASELINE.POOL_TYPE "avg"\
     DATASETS.DATA_PATH "('${DATA_DIR}')" DATASETS.TRAIN_PATH "rep_trainVal2"\
     MODEL.PRETRAIN_PATH "('${PRETRAIN}')"  \
     OUTPUT_DIR "('${SAVE_DIR}')" \
     TEST.WEIGHT "${WEIGHT}"
```
then you can get pseudo data in pseudo_savepath

### step2 copy pseudo data to trainVal data
#### modify prepare_pseudo.py
```python
    root_dir = '/data/Dataset/PReID/'  # root of dataset
    
    origin_path = root_dir+'dataset2/'+'pid_dataset/' # original trainval data
    pseudo_path = '../rep_work_dirs/testb_pseudo_hist_065_080/' # pseudo data

    save_dir = root_dir+'testb_pseudo_hist_065_080_dataset/' # save path
```
#### run prepare_pseudo.py

### step3 train the model again using mix dataset(original trainval data and pseudo data)

## Model Ensemble
### step1 modify dist_ensemble.py
```python
 save_dir = '../rep_work_dirs/testb_ensembles/'  # save path
 query_dir='/data/Dataset/PReID/dataset2/query_/' # query path
 gallery_dir='/data/Dataset/PReID/dataset2/gallery_/' # gallery path
 save_fname = 'ensemble1.json' # submit filename
 dist_fnames = [  ## distance matrix
    #     'origin_tpl03_e90_flip_sub_aqe.pkl',
    #     'origin_tpl03_e80_flip_sub_aqe.pkl',
    #     'finetune_tpl05_e16_flip_sub_aqe.pkl',
    #     'finetune_tpl05_e14_flip_sub_aqe.pkl'

    # ]
```

### step2 run dist_ensemble.py

## Notes:
1.Due to time constraints, the best solution we eventually adopted was to use a baseline model and a pseudo-labeled baseline model for fusion.  
&nbsp;  
2.We also trained MGN to a better effect, but the final effect of mgn is not as good as the baseline + arcface + cj.  
&nbsp;  
3.The MFN network comes from one of my teammates, and he plans to use the model in a paper. So it cannot be open source now. If you are interested, you can follow his Github：https://github.com/douzi0248/Re-ID  
&nbsp;  
4.The codes are expanded on a ReID-baseline , which is open sourced by Hao Luo.(Thanks for [Hao Luo](https://github.com/michuanhaohao "Hao Luo") and [DTennant](https://github.com/DTennant "DTennant"), our baseline model comes from https://github.com/michuanhaohao/reid-strong-baseline and https://github.com/DTennant/reid_baseline_with_syncbn)
