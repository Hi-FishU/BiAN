import torch.utils.data as data
import cv2
import os
from glob import glob
import torch
import random
import numpy as np
import json
import h5py
from utils import Model_Logger, random_segmentation


IMG_EXTENSIONS = ['*.png','*.jpeg', '*.jpg', '*.tif', '*.PNG', '*.JPEG', '*.JPG', '*.TIF']

logger = Model_Logger('data')


def random_crop(img_height, img_width, crop_height, crop_width):
    res_height = img_height - crop_height
    res_width = img_width - crop_width
    i = random.randint(0, res_height)
    j = random.randint(0, res_width)
    return i, j, crop_height, crop_width

def get_image_filename_list(root) -> list:

    if os.path.exists(root) is not True:
        raise FileNotFoundError("{} does not exist.".format(root))

    img_dir = os.path.join(root, "raw")
    dot_dir = os.path.join(root, "dot")
    if os.path.exists(img_dir) is not True or os.path.exists(dot_dir) is not True:
        raise Exception("Unknown dataset structure {}".format(root))
    # Validate the file structure
    try:
        json_file = os.path.join(root, glob("*.json", root_dir=root)[0])
    except Exception:
        raise FileNotFoundError("Cannot find json file of filenames in {}".format(root))

    with open(json_file, 'r') as f:
        image_file_list = json.load(f)

    return image_file_list

def train_val(im_list, ratio=0.9):
    N = int(float(len(im_list))*ratio)
    idx = torch.randperm(len(im_list))
    train_list = [im_list[i] for i in idx[0:N]]
    val_list = [im_list[i] for i in idx[N+1:]]
    return train_list, val_list

# Implement Dataset Class
class Cell_dataset(data.Dataset):
    def __init__(self, dataset_dictionary:str,
                 type:str, crop_size:tuple | None = None,
                 resize: int | None = None,
                 training_factor:int = 100):

        if type not in ['h5py', 'image']:
            raise TypeError("{} is not yet supported.".format(type))

        self.root = dataset_dictionary
        self.C_size = crop_size
        self.R_size = resize
        self.type = type
        self.mode = None
        self.factor = training_factor
        self.load_data()


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.copy(self.imgs[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dot = np.copy(self.dots[index])
        dot = cv2.cvtColor(dot, cv2.COLOR_BGR2GRAY)

        w, h = img.shape

        if self.R_size is not None:
            img_max = max(w, h)
            ratio = self.R_size / float(img_max)
            w = int(float(w) * ratio)
            h = int(float(h) * ratio)
            img = np.resize(img, (w, h))
            dot = np.resize(dot, (w, h))

        # Switching between Train mode and Validation mode
        if self.C_size is not None and self.mode == 'train':
            C_w, C_h = self.C_size
            i, j, C_w, C_h = random_crop(w, h, C_w, C_h)
            img = img[j: j + C_h, i: i + C_w]
            dot = dot[j: j + C_h, i: i + C_w]
        dot[dot != 0] = 1
        count = np.sum(np.copy(dot)).astype(np.int64)
        dot = cv2.GaussianBlur(dot * self.factor, (5, 5), sigmaX=0)
        img = img.astype(np.int32)
        dot = dot.astype(np.int32)
        rand_mask = random_segmentation(img.shape)

        return torch.from_numpy(img).float().unsqueeze(0), \
            torch.from_numpy(dot).float().unsqueeze(0), \
                rand_mask(dot).float().unsqueeze(0), count


    def load_data(self):
        if self.type == 'h5py':
            try:
                filename = glob(pathname = '*.hdf5', root_dir = self.root)[0]
            except IndexError:
                raise FileNotFoundError("hdf5 file not found in {}. Check dataset type and file.".format(self.root))
            h5_file = h5py.File(os.path.join(self.root, filename))
            try:
                self.imgs = np.asarray(h5_file['imgs'])
                self.dots = np.asarray(h5_file['counts'])
            except KeyError:
                raise KeyError("Not a proper hdf5 structure. It should include \'imgs\' key and \'counts\' key!")
        else: # type == 'image'
            file_list = get_image_filename_list(self.root)
            if file_list is None:
                raise FileNotFoundError("There are not any supported image files in {}".format(self.root))
            imgs = []
            dots = []
            for dot_filename, raw_filename in file_list:
                dot_filename = os.path.join(self.root, 'dot', dot_filename)
                raw_filename = os.path.join(self.root, 'raw', raw_filename)
                if os.path.exists(raw_filename) is not True:
                    logger.warning("Could not find the annotations file {}. Skipping this file".format(raw_filename))
                    continue
                if os.path.exists(dot_filename) is not True:
                    logger.warning("Could not find the annotations file {}. Skipping this file".format(dot_filename))
                    continue
                imgs.append(np.asarray(cv2.imread(raw_filename)))
                dots.append(np.asarray(cv2.imread(dot_filename)))
            self.imgs = imgs
            self.dots = dots

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'

