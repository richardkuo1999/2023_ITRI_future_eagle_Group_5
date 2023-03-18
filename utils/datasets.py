import os
import cv2
import yaml
import glob
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from prefetch_generator import BackgroundGenerator

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from utils.general import one_hot_it_v11_dice
from utils.augmentations import augment_hsv, random_perspective, letterbox,\
                                 letterbox_for_img



def create_dataloader(args, hyp, data_dict, batch_size, normalize, is_train=True, shuffle=True):
    normalize = transforms.Normalize(
            normalize['mean'], normalize['std']
        )
    
    datasets = eval('BddDataset')(
        args=args,
        hyp=hyp,
        data_dict=data_dict,
        dataSet=data_dict['train'] if is_train else data_dict['val'],
        is_train=is_train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    loader = DataLoaderX(
        datasets,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=AutoDriveDataset.collate_fn
    )
    return loader, datasets

class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """
    def __init__(self, args, hyp, data_dict, dataSet, is_train, transform=None):
        """
        initial all the characteristic

        Inputs:
        -args: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.hyp = hyp
        self.data_dict = data_dict
        self.transform = transform
        self.inputsize = args.img_size

        self.Tensor = transforms.ToTensor()

        # Data Root
        self.img_root = Path(dataSet[0])
        self.mask_root = Path(dataSet[1])
        self.label_info = data_dict['class_name']

        self.mask_list = self.mask_root.iterdir()

        self.db = []

    
    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, args, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and ground-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        hyp = self.hyp
        data = self.db[idx]
        resized_shape = max(self.inputsize) if isinstance(self.inputsize, list) \
                                            else self.inputsize

        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]  # orig hw

        label = cv2.imread(data["label"])


        #resize
        (img, label), ratio, pad = letterbox((img, label),\
                                         resized_shape, auto=True, scaleup=self.is_train)
        h, w = img.shape[:2]

        
        if self.is_train:
            combination = (img, label)
            (img, label) = random_perspective(
                combination=combination,
                degrees=hyp['rot_factor'],
                translate=hyp['translate'],
                scale=hyp['scale_factor'],
                shear=hyp['shear']
            )
            #print(labels.shape)
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            # img, drivable_label, labels = cutout(combination=combination, labels=labels)

        if self.is_train:
        # random left-right flip
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                label = np.fliplr(label)

            # random up-down flip
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                label = np.flipud(label)

        

        img = np.ascontiguousarray(img)
    
        
        
        

        label = one_hot_it_v11_dice(label, self.label_info)

        # from PIL import Image
        # aaa = img.copy()
        # label_bool = label.copy().astype(dtype=bool)
        # for i in range(0,len(label_bool[0,0])):
        #     aaa[label_bool[:,:,i]] = self.label_info[list(self.label_info)[i]][:3]

        # aaa = Image.fromarray(aaa, "RGB")
        # aaa.save(f'{idx}.bmp')

        label = self.Tensor(label)


        img = self.transform(img)
        target = label
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        return img, target, data["image"], shapes

    def filter_data(self, db):
        """
        finished on children dataset
        """
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes= zip(*batch)
        label_seg = []
        for i, l in enumerate(label):
            l_seg = l
            label_seg.append(l_seg)
        return torch.stack(img, 0), torch.stack(label_seg, 0), paths, shapes

class BddDataset(AutoDriveDataset):
    def __init__(self, args, hyp, data_dict, dataSet, is_train, transform=None):
        super().__init__(args, hyp, data_dict, dataSet, is_train, transform)
        self.db = self.__get_db()
        
    def __get_db(self):
        """get database from the annotation file

        Returns:
            gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'label':, 'mask':,'lane':}
            image: image path
            label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
            mask: path of the driver area segmentation label path
            lane: path of the lane segmentation label path
        """
        gt_db = []
        for mask in tqdm(list(self.mask_list)):
            mask_path = str(mask)
            image_path = mask_path.replace(str(self.mask_root), 
                                str(self.img_root))

            rec = [{
                'image': image_path,
                'label': mask_path,
            }]

            gt_db += rec
        return gt_db




img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']
class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
            h0, w0 = img0.shape[:2]

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ')
            h0, w0 = img0.shape[:2]

        # Padded resize
        img, ratio, pad = letterbox_for_img(img0, new_shape=self.img_size, auto=True)
        h, w = img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # Convert
        #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)


        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap, shapes

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


