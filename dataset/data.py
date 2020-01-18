import torch
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T
import glob
import re
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img_type = 'RGB'
            img = Image.open(img_path).convert(img_type)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. "
                            "Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def pil_simple_hist_predictor(image,channel=2,thres=100): #RGB; by the first channel
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) #BGR
    hist = cv2.calcHist([image], [channel], None, [256], [0, 256]) #绘制各个通道的直方图
    if hist[0]>thres:
        return 0
    else:
        return 1
class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, cfg, transform=None):
        self.dataset = dataset
        self.cfg = cfg
        self.transform = transform
        self.use_histlabel = cfg.DATASETS.HIST_LABEL.USE
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.use_histlabel:
            histlabel = pil_simple_hist_predictor(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.use_histlabel:
            return img, pid, camid, img_path , histlabel
        return img, pid, camid, img_path
        
class SepNormImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, cfg, transform0=None, transform1=None):
        self.dataset = dataset
        self.cfg = cfg
        self.transform0 = transform0
        self.transform1 = transform1
        # self.l0cnt = 0
        # self.l1cnt = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
   
        histlabel = pil_simple_hist_predictor(img)
        if self.transform0 is not None:
            if histlabel:
                img = self.transform1(img)
                # self.l1cnt += 1
            else:
                img = self.transform0(img)
                # self.l0cnt += 1
        # if self.l1cnt % 100==0: 
            # print(self.l0cnt,self.l1cnt)
        return img, pid, camid, img_path

class BaseDataset:
    def __init__(self, root='/home/zbc/data/reid',
                    train_dir='', query_dir='', gallery_dir='',
                    verbose=True,exemplar_cfg=None, **kwargs):
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, train_dir)
        self.query_dir = osp.join(self.dataset_dir, query_dir)
        self.gallery_dir = osp.join(self.dataset_dir, gallery_dir)
        self.exemplar_cfg = exemplar_cfg
        if self.exemplar_cfg.USE:
            self.exemplar_dir = osp.join(self.dataset_dir, exemplar_cfg.PATH)

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        if self.exemplar_cfg.USE:
            img_paths = glob.glob(osp.join(self.exemplar_dir, '*.png'))
            exemplar = []
            #         
            # pattern = re.compile(r'\d+_c\d+')
            # pid_container = set()
            # for img_path in img_paths:
            #     pid = pattern.findall(img_path)[0]
            #     pid_container.add(pid)
            # pid2label = {pid: label for label, pid in enumerate(pid_container)}
            # for cid,img_path in enumerate(img_paths):
            #     pid = pattern.findall(img_path)[0]
            #     exemplar.append((img_path, pid2label[pid], cid))
            for pid,img_path in enumerate(img_paths):
                exemplar.append((img_path, pid, pid))
            self.exemplar = exemplar



        if verbose:
            print("=> Data loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        if self.exemplar_cfg.USE:
            self.num_exemplar_pids, self.num_exemplar_imgs, self.num_exemplar_cams = self.get_imagedata_info(self.exemplar)

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        if self.exemplar_cfg.USE:
            num_exemplar_pids, num_exemplar_imgs, num_exemplar_cams = self.get_imagedata_info(self.exemplar)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        if self.exemplar_cfg.USE:
            print("  exemplar | {:5d} | {:8d} | {:9d}".format(num_exemplar_pids, num_exemplar_imgs, num_exemplar_cams))
        print("  ----------------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if self.exemplar_cfg.USE:
            if not osp.exists(self.exemplar_dir):
                raise RuntimeError("'{}' is not available".format(self.exemplar_dir))
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset



def init_dataset(cfg):
    """
        Use path in cfg to init a dataset
        the dataset should be the following format
        - Each Image should be named in 
                (pid)_c(camid)_(iid).jpg
            where pid is the person id, 
                  camid is camera id,
                  iid is image id(unique to every image)
        - train set and val set should be organzed as
            cfg.DATASETS.TRAIN_PATH: all the training images
            cfg.DATASETS.QUERY_PATH: all the query images
            cfg.DATASETS.GALLERY_PATH: all the gallery images
    """
    return BaseDataset(root=cfg.DATASETS.DATA_PATH, train_dir=cfg.DATASETS.TRAIN_PATH,
                    query_dir=cfg.DATASETS.QUERY_PATH, gallery_dir=cfg.DATASETS.GALLERY_PATH,exemplar_cfg=cfg.DATASETS.EXEMPLAR)

