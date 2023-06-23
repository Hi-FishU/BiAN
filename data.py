import torch.utils.data as data
import cv2
import os
from glob import glob
import torch
import random
import numpy as np
import json
import h5py
from utils import Model_Logger, random_segmentation, _setup_size
from torchvision import transforms


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
    def __init__(self, root:str,
                 type:str, crop_size:tuple | None = None,
                 transform: bool = True,
                 resize:tuple = None,
                 training_factor:int = 100):

        if type not in ['h5py', 'image']:
            raise TypeError("{} is not yet supported.".format(type))

        self.root = root
        self.C_size = _setup_size(resize, 'Error random crop size.')
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
        dot = cv2.cvtColor(dot, cv2.COLOR_BGR2GRAY)

        if self.resize:
            img = np.resize(img, self.resize)
            dot = np.resize(dot, self.resize)

        img = img.astype(np.int32)
        dot[dot != 0] = 1
        count = np.sum(np.copy(dot)).astype(np.int64)
        dot = cv2.GaussianBlur(dot * self.factor, (5, 5), sigmaX=0)
        dot = dot.astype(np.int32)
        img = torch.from_numpy(img).float()
        dot = torch.from_numpy(dot).float()



        if self.transform and self.mode == 'train':
            global_transform = transforms.Compose([
                transforms.RandomCrop(self.C_size),
            ])
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])
            img = global_transform(img)
            dot = global_transform(dot)
            img = train_transform(img)



        return img.unsqueeze(0), dot.unsqueeze(0), count


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

class MNIST_dataset(data.Dataset):
    def __init__(self, dataset_dictionary:str,
                 filelist:list[str],
                 transform:tuple | None = None):
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
            img_filename, label= filename.split(' ')
            img = np.asarray(cv2.imread(os.path.join(self.root, img_dir, img_filename)))
            if self.resize is not None:
                img = np.resize(img, (self.resize, self.resize))
            self.imgs.append(img)
            self.labels.append(np.asarray([int(label)]))

