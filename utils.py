import datetime
import torch.nn as nn
import logging
import sys
import os
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

class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, n = 1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def get(self, key):
        result = {'sum': self.sum, 'count': self.count, 'avg': self.avg}
        return result[key]


# Logging out module
class Model_Logger(logging.Logger):
    def __init__(self, name, level = logging.INFO):
        super(Model_Logger, self).__init__(name, level)
        self.name = name
        self.level = level
        time_strip = datetime.datetime.now()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # logging basic settings

        fileHandler = LocalFileHandler(
            filename=os.path.join(Constants.LOG_FOLDER, "{}.log".format(time_strip)),
            mode='a')
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.DEBUG)
        # File handler

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(self.level)
        # Stream handler

        self.addHandler(fileHandler)
        self.addHandler(streamHandler)
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





