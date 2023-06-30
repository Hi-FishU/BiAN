import argparse
import datetime
import os



class ArgParser(argparse.ArgumentParser):

    def load_arguments(self):
        self.add_argument('-T', '--title', type=str, default='Exp without separated infill.',)

        self.add_argument('-L', '--learning-rate', type=float, default=1e-3)
        self.add_argument('-E', '--epoch', type=int, default=800)
        self.add_argument('-N', '--device', type=str, default='0')

        self.add_argument('-LD','--lr-decay', type=int, default=0.1)
        # self.add_argument('-M', '--momentum', type=float, default=0.9,
        #                   help='Momentum value for SGD optimizer.')
        self.add_argument('-WD', '--weight-decay', type=float, default=1e-3,
                          help="Decay weight for optimizer.")
        self.add_argument('-RT', '--restart-step', type=int, default=50,
                          help='Restart step for cosine annealing warm restarts.')

        self.add_argument('-BM', '--BN-momentum', type=float, default=0.9,
                          help="Momentum for batch normalize.")
        self.add_argument('-DP', '--dropout', type=float, default=0.1)
        self.add_argument('-O', '--output', type=str, default='')

        self.add_argument('-DS', '--source-dataset', type=str, default='vgg')
        self.add_argument('-DT', '--target-dataset', type=str, default='adi')
        self.add_argument('-TS', '--source-dataset-type', type=str, default='h5py')
        self.add_argument('-TT', '--target-dataset-type', type=str, default='h5py')
        self.add_argument('-TR', '--training-ratio', type=float, default=0.5,
                           help="Training data ratio, test ratio set as 0.1")
        self.add_argument('-B', '--batch-size', type=int, default=64)
        self.add_argument('-TSS', '--training-scale_s', type=int, default=100,
                          help="mbm: 1000, dcc: 500,\
                          adi: 100, vgg: 100, 'mbc: 1000")
        self.add_argument('-TST', '--training-scale_t', type=int, default=100,
                          help="mbm: 1000, dcc: 500,\
                          adi: 100, vgg: 100, 'mbc: 1000")
        self.add_argument('-RS', '--image-resize', type=int, default=96)
        self.add_argument('-P', '--patch-size', type=tuple, default=64,
                          help="Cropping size of image.")
        self.add_argument('-WS', '--warm-start', type=int, default=150,
                          help="Epochs only train regressor on source domain.")

class Constants:

    ROOT_PATH = '/home/zhuonan/code/NewModel' # Baseline directory path
    MODEL_NAME = 'UDA' # Model directory path

    DATA_FOLDER = os.path.join(ROOT_PATH, 'data')
    OUTPUT_FOLDER = os.path.join(ROOT_PATH, 'output')
    LOG_FOLDER = os.path.join(ROOT_PATH, 'log')
    LOG_NAME = datetime.datetime.now()
    TARGET_TRAIN_FILELIST = ''
    TARGET_VALID_FILELIST = ''

    DATASET = {'vgg': 'VGG', 'mbm':'MBM', 'adi':'ADI', 'mnist':'MNIST','mnist_m':'MNIST_moving', 'dcc':'DCC'}

    CFG = [[32, 'R', 'M', 64, 'R', 'M', 128, 'R', 'M', 512, 'R'], [128, 'R', 'U', 64, 'R', 'U', '32', 'R', 'U', 1, 'R']]


