import datetime
import torch
import cv2
import os
from torch.utils.data import DataLoader, random_split
from data import Cell_dataset
from config import ArgParser, Constants
from model import FCRN
from utils import Model_Logger, layer_maker, AverageMeter
import time
from torch.utils.tensorboard import SummaryWriter


logger = Model_Logger('train')
logger.enable_exception_hook()
writer = SummaryWriter()


def train(args):
    # Initialize device
    logger.info("Training date time: {}".format(datetime.datetime.now()))
    start_time = time.time()
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.device))
        logger.info("Using device: CUDA_{}".format(args.device))
    else:
        logger.warn("Using device: CPU")

    logger.info("Start loading data.")
    dataset = Cell_dataset(os.path.join(Constants.DATA_FOLDER, Constants.DATASET[args.dataset]),
                           type=args.dataset_type,
                           crop_size=args.patch_size)
    valid_ratio = 1 - args.training_ratio - 0.1
    train_set, valid_set, test_set = random_split(dataset, [args.training_ratio, valid_ratio, 0.1])
    train_set.dataset.train()
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    logger.info("Loading data completed. Elapsed time: {:.2f}sec.".format(time.time() - start_time))

    logger.info("Start initailizing model")
    layers = layer_maker(Constants.CFG)
    model = FCRN(layers)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss(reduction='sum')
    count_mse = torch.nn.MSELoss(reduction='mean')
    count_mae = torch.nn.L1Loss(reduction='mean')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=5,
                                                gamma=args.lr_decay)
    logger.info("Initialization Completed. Elapsed time: {:.2f}sec".format(time.time() - start_time))

    #TODO Checkpoint loading

    train_time = time.time()
    for epoch in range(args.epoch):
        epoch_time = time.time()
        logger.info("Epoch: {}".format(epoch + 1))

        epoch_loss = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_mae = AverageMeter()
        valid_loss = AverageMeter()
        valid_mse = AverageMeter()
        valid_mae = AverageMeter()
        test_loss = AverageMeter()
        test_mse = AverageMeter()
        test_mae = AverageMeter()


        model.train()

        for index, (img, dot, ground_truth) in enumerate(train_dataloader):
            img = img.to(device)
            dot = dot.to(device)

            with torch.set_grad_enabled(True):
                outputs = model(img)
                dot = torch.resize_as_(dot, outputs)
                if torch.sum(dot) == 0:
                    logger.info("Step {} of epoch {} has the zero cells annotation.".format(index, epoch))
                    continue
                loss = criterion(outputs, dot / 100)
            # TODO Dot blurring. Outputs remain zero
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counts = outputs.sum([1, 2, 3]).detach().cpu()
                res = abs(torch.mean(counts - ground_truth))

            epoch_loss.update(loss.item() / args.batch_size)
            epoch_mse.update(count_mse(counts, ground_truth))
            epoch_mae.update(count_mae(counts, ground_truth))

            if index % 100 == 0:
                logger.info("Error: {:.2f}, Pred: {:.2f}, Gt: {:.2f}, Loss: {:.3f}".format(res, counts.sum(), ground_truth.sum(), loss.item()))
                writer.add_scalar(tag='Running loss', scalar_value=loss.item(), global_step=epoch)

        # Validation
        model.eval()
        for _, (img, dot, ground_truth) in enumerate(valid_dataloader):
            img = img.to(device)
            dot = dot.to(device)

            outputs = model(img)
            dot = torch.resize_as_(dot, outputs)

            loss = criterion(outputs, dot)
            loss = loss / args.batch_size

            counts = outputs.sum().detach().cpu()
            res = torch.mean(counts - ground_truth)

            valid_loss.update(loss.item())
            valid_mse.update(count_mse(counts, ground_truth))
            valid_mae.update(count_mae(counts, ground_truth))

        # TODO Implement writer

        logger.info("Epoch:{} \nTrain\
                    Loss: {:.2f}, \
                    MSE: {:.2f}, \
                    MAE: {:.2f}, \
                    \nValid\
                    Loss: {:.2f}, \
                    MSE: {:.2f}, \
                    MAE: {:.2f}, \
                    \nCost: {:.1f} sec".format(epoch + 1,
                                                epoch_loss.get('avg'),
                                                epoch_mse.get('avg'),
                                                epoch_mae.get('avg'),
                                                valid_loss.get('avg'),
                                                valid_mse.get('avg'),
                                                valid_mae.get('avg'),
                                                time.time() - epoch_time))
        scheduler.step()

    logger.info("Training completed, starting testing...")
    model.eval()

    for _, (img, dot, ground_truth) in enumerate(test_dataloader):
            img = img.to(device)
            dot = dot.to(device)

            outputs = model(img)
            dot = torch.resize_as_(dot, outputs)

            loss = criterion(outputs, dot)
            loss = loss / args.batch_size

            counts = outputs.sum().detach().cpu()
            res = torch.mean(counts - ground_truth)

            test_loss.update(loss.item())
            test_mse.update(count_mse(counts, ground_truth))
            test_mae.update(count_mae(counts, ground_truth))
    logger.info("Test\
                Loss: {:.2f}, \
                MSE: {:.2f}, \
                MAE: {:.2f}, \
                \nCost: {:.1f} sec".format(epoch,
                                            test_loss.get('avg'),
                                            test_mse.get('avg'),
                                            test_mae.get('avg'),
                                            time.time() - start_time))
    writer.close()
    logger.info("Finished.")

if __name__ == '__main__':
    parser = ArgParser()
    parser.load_arguments()
    args = parser.parse_args()
    train(args)
