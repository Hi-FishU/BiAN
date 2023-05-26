import argparse
import os



class ArgParser(argparse.ArgumentParser):

    def load_arguments(self):
        self.add_argument('-L', '--learning-rate', type=float, default=1e-2)
        self.add_argument('-E', '--epoch', type=int, default=100)
        self.add_argument('-N', '--device', type=str, default='0')

        self.add_argument('-LD','--lr-decay', type=int, default=0.1)
        self.add_argument('-M', '--momentum', type=float, default=0.9,
                          help='Momentum value for SGD optimizer.')
        self.add_argument('-WD', '--weight-decay', type=float, default=5e-4,
                          help="Decay weight for SGD optimizer.")
        self.add_argument('-O', '--output', type=str, default='')

        self.add_argument('-D', '--dataset', type=str, default='vgg')
        self.add_argument('-T', '--dataset-type', type=str, default='image')
        self.add_argument('-R', '--training-ratio', type=float, default=0.7,
                           help="Training data ratio, test ratio set as 0.1")
        self.add_argument('-B', '--batch-size', type=int, default=8)
        self.add_argument('-G', '--Gaussian-scale', type=int, default=100)
        self.add_argument('-P', '--patch-size', type=tuple, default=None,
                          help="Cropping size of image.")

class Constants:

    ROOT_PATH = '/home/zhuonan/code/baselines' # Baseline directory path
    MODEL_NAME = 'FCRN' # Model directory path

    DATA_FOLDER = os.path.join(ROOT_PATH, 'data')
    OUTPUT_FOLDER = os.path.join(ROOT_PATH, MODEL_NAME, 'output')
    LOG_FOLDER = os.path.join(ROOT_PATH, MODEL_NAME, 'log')


    DATASET = {'vgg': 'VGG', 'mbm':'MBM', 'adi':'ADI'}

    CFG = [[32, 'R', 'M', 64, 'R', 'M', 128, 'R', 'M', 512, 'R'], [128, 'R', 'U', 64, 'R', 'U', '32', 'R', 'U', 1, 'R']]
    # CFG = [[32, 64, 'M', 128, 256, 'M'], [256, 'U', 256, 'U', 1]]
