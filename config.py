import argparse
import datetime
import os



class ArgParser(argparse.ArgumentParser):

    def load_arguments(self):
        self.add_argument('-L', '--learning-rate', type=float, default=1e-4)
        self.add_argument('-E', '--epoch', type=int, default=350)
        self.add_argument('-N', '--device', type=str, default='0')

        self.add_argument('-LD','--lr-decay', type=int, default=0.1)
        # self.add_argument('-M', '--momentum', type=float, default=0.9,
        #                   help='Momentum value for SGD optimizer.')
        self.add_argument('-WD', '--weight-decay', type=float, default=1e-4,
                          help="Decay weight for optimizer.")
        self.add_argument('-RT', '--restart-step', type=int, default=50,
                          help='Restart step for cosine annealing warm restarts.')

        self.add_argument('-BM', '--BN-momentum', type=float, default=0.9,
                          help="Momentum for batch normalize.")
        self.add_argument('-DP', '--dropout', type=float, default=0.1)
        self.add_argument('-O', '--output', type=str, default='')

        self.add_argument('-D', '--dataset', type=str, default='mbm')
        self.add_argument('-T', '--dataset-type', type=str, default='image')
        self.add_argument('-TR', '--training-ratio', type=float, default=0.7,
                           help="Training data ratio, test ratio set as 0.1")
        self.add_argument('-B', '--batch-size', type=int, default=64)
        self.add_argument('-S', '--training-scale', type=int, default=1000,
                          help="mbm: 1000, dcc: 500,\
                          adi: 100, vgg: 100, 'mbc: 1000")
        self.add_argument('-P', '--patch-size', type=tuple, default=(256, 256),
                          help="Cropping size of image.")

class Constants:

    ROOT_PATH = '/home/zhuonan/code/baselines' # Baseline directory path
    MODEL_NAME = 'FCRN' # Model directory path

    DATA_FOLDER = os.path.join(ROOT_PATH, 'data')
    OUTPUT_FOLDER = os.path.join(ROOT_PATH, MODEL_NAME, 'output')
    LOG_FOLDER = os.path.join(ROOT_PATH, MODEL_NAME, 'log')
    LOG_NAME = datetime.datetime.now()

    DATASET = {'vgg': 'VGG', 'mbm':'MBM', 'adi':'ADI'}

    CFG = [[32, 'R', 'M', 64, 'R', 'M', 128, 'R', 'M', 512, 'R'], [128, 'R', 'U', 64, 'R', 'U', '32', 'R', 'U', 1, 'R']]
    # CFG = [[32, 64, 'M', 128, 256, 'M'], [256, 'U', 256, 'U', 1]]
