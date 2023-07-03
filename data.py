import os
from glob import glob

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from utils import (Model_Logger, RandomRotation90, _setup_size,
                   get_image_filename_list, random_crop, random_segmentation)

IMG_EXTENSIONS = [
    '*.png', '*.jpeg', '*.jpg', '*.tif', '*.PNG', '*.JPEG', '*.JPG', '*.TIF'
]

logger = Model_Logger('data')


# Implement Dataset Class
class Cell_dataset(data.Dataset):
    def __init__(self,
                 root: str,
                 type: str,
                 crop_size: tuple | None = None,
                 transform: bool = True,
                 resize: tuple = None,
                 training_factor: int = 100):

        if type not in ['h5py', 'image']:
            raise TypeError("{} is not yet supported.".format(type))

        self.root = root
        self.C_size = _setup_size(crop_size, 'Error random crop size.')
        self.resize = _setup_size(resize, 'Error resize size.')
        self.type = type
        self.mode = None
        self.factor = training_factor
        self.transform = transform
        self.load_data()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.copy(self.imgs[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dot = np.copy(self.dots[index])
        if len(dot.shape) < 3:
            dot = np.expand_dims(dot, -1)
        if dot.shape[-1] != 1:
            dot = cv2.cvtColor(dot, cv2.COLOR_BGR2GRAY)
        dot = dot.squeeze(-1)
        if self.resize:
            img = np.resize(img, self.resize)
            dot = np.resize(dot, self.resize)
        # dot = cv2.GaussianBlur(dot, (5, 5), sigmaX=0)
        img = img.astype(np.float32)
        dot = dot.astype(np.float32)
        img = torch.Tensor(img)
        dot = torch.Tensor(dot)

        if self.transform and self.mode == 'train':
            img_dot = torch.concat(
                [img.unsqueeze(-1), dot.unsqueeze(-1)], dim=-1).permute(2, 0, 1)

            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                RandomRotation90(90),
            ])
            img_dot = transform(img_dot).permute(1, 2, 0)
            img = img_dot[:, :, 0]
            dot = img_dot[:, :, 1]

        # Random cropping
        if self.C_size:
            i, j, height, width = random_crop(img.shape, self.C_size)
            img = transforms.functional.crop(img, i, j, height, width)
            dot = transforms.functional.crop(dot, i, j, height, width)


        # count = torch.sum(dot).int()
        # dot = dot * self.factor
        return img.unsqueeze(0), dot.unsqueeze(0)

    def load_data(self):
        if self.type == 'h5py':
            try:
                filename = glob(pathname='*.hdf5', root_dir=self.root)[0]
            except IndexError:
                raise FileNotFoundError(
                    "hdf5 file not found in {}. Check dataset type and file.".
                    format(self.root))
            h5_file = h5py.File(os.path.join(self.root, filename))
            try:
                self.imgs = np.asarray(h5_file['imgs'])
                self.dots = np.asarray(h5_file['counts'])
            except KeyError:
                raise KeyError(
                    "Not a proper hdf5 structure. It should include \'imgs\' key and \'counts\' key!"
                )
        else:  # type == 'image'
            file_list = get_image_filename_list(self.root)
            if file_list is None:
                raise FileNotFoundError(
                    "There are not any supported image files in {}".format(
                        self.root))
            imgs = []
            dots = []
            for dot_filename, raw_filename in file_list:
                dot_filename = os.path.join(self.root, 'dot', dot_filename)
                raw_filename = os.path.join(self.root, 'raw', raw_filename)
                if os.path.exists(raw_filename) is not True:
                    logger.warning(
                        "Could not find the annotations file {}. Skipping this file"
                        .format(raw_filename))
                    continue
                if os.path.exists(dot_filename) is not True:
                    logger.warning(
                        "Could not find the annotations file {}. Skipping this file"
                        .format(dot_filename))
                    continue
                imgs.append(np.asarray(cv2.imread(raw_filename)))
                dots.append(np.asarray(cv2.imread(dot_filename)))
            self.imgs = imgs
            self.dots = dots

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'


class MNIST_dataset(data.Dataset):
    def __init__(self,
                 dataset_dictionary: str,
                 filelist: list[str],
                 transform: tuple | None = None):
        self.root = dataset_dictionary
        self.transform = transform
        self.filenames = filelist
        self.load_data()

    def __len__(self):
        return self.labels.__len__()

    def __getitem__(self, index):
        img = np.copy(self.imgs[index])
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)

        return img.float(), torch.from_numpy(label).long()

    def load_data(self):
        self.imgs = []
        self.labels = []
        with open(self.filenames, 'r') as f:
            filelist = f.readlines()
        img_dir = os.path.basename(self.filenames)
        for filename in filelist:
            img_filename, label = filename.split(' ')
            img = np.asarray(
                cv2.imread(os.path.join(self.root, img_dir, img_filename)))
            if self.resize is not None:
                img = np.resize(img, (self.resize, self.resize))
            self.imgs.append(img)
            self.labels.append(np.asarray([int(label)]))
