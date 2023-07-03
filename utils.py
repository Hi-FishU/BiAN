import json
import logging
import numbers
import os
import random
import sys
from collections.abc import Sequence
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from config import Constants


def layer_maker(cfg, in_channels=1, conv_kernel_size=3, up_kernel_size=3, batch_norm=False, dilation=1):
    # Make the convolution layers
    layers = []
    for v in cfg[0]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U':
            layers += [nn.Upsample(scale_factor=2, mode='bilinear')]
        elif v == 'R':
            layers += [nn.ReLU(inplace=True)]
        elif isinstance(v, int):
            conv2d = nn.Conv2d(in_channels, v,
                               kernel_size=conv_kernel_size,
                               padding=dilation,
                               dilation=dilation)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v)]
            else:
                layers += [conv2d]
            in_channels = v

    for v in cfg[1]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U':
            layers += [nn.Upsample(scale_factor=2, mode='bilinear')]
        elif v == 'R':
            layers += [nn.ReLU(inplace=True)]
        elif isinstance(v, int):
            conv2d = nn.Conv2d(in_channels, v, kernel_size=up_kernel_size, padding=dilation, dilation=dilation)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v)]
            else:
                layers += [conv2d]

            in_channels = v

    return nn.Sequential(*layers)

def random_segmentation(size):
    rand_mask = np.random.normal(0, 1, size)
    rand_mask[np.where(rand_mask > 0)] = 1
    rand_mask[np.where(rand_mask < 0)] = 0
    return rand_mask


class RandomRotation90(transforms.RandomRotation):
    def get_params(self, img):
        # Choose a random rotation angle in multiples of 90 degrees
        angle = random.randint(0, 3) * 90

        # Return the rotation angle as a tuple
        return angle

def random_crop(img_size, crop_size):
    img_height = img_size[0]
    img_width = img_size[1]
    crop_height = crop_size[0]
    crop_width = crop_size[1]
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

@staticmethod
def _setup_size(size, error_msg):
    if size is None:
        return None

    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.min = 1e+4

    def update(self, value, n = 1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.min = min(self.min, value)

    def get(self, key):
        result = {'sum': self.sum, 'count': self.count, 'avg': self.avg, 'min': self.min}
        return result[key]


# Logging out module
class Model_Logger(logging.Logger):

    _instance = None

    def __init__(self, name, level = logging.INFO):
        super(Model_Logger, self).__init__(name, level)
        self.name = name
        self.level = level
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # logging basic settings

        fileHandler = LocalFileHandler(
            filename=os.path.join(Constants.LOG_FOLDER, "{}.log".format(Constants.LOG_NAME)),
            mode='a+')
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.DEBUG)
        # File handler

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(self.level)
        # Stream handler

        self.addHandler(fileHandler)
        self.addHandler(streamHandler)
        self.enable_exception_hook()
        # Add handler

    def exception_hook(self, exc_type, exc_value, exc_traceback):
        self.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    def enable_exception_hook(self):
        sys.excepthook = self.exception_hook
        # Set the exception hook


class LocalFileHandler(logging.FileHandler):
    def __init__(self, filename: str,
                 mode: str = "a",
                 encoding: str | None = None,
                 delay: bool = False,
                 errors: str | None = None) -> None:
        super().__init__(filename, mode, encoding, delay, errors)


