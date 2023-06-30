import datetime
import os
import time
import itertools
import numpy as np

import torch
from config import ArgParser, Constants
from model import UDACounting
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from utils import AverageMeter, Model_Logger

from data import Cell_dataset

logger = Model_Logger('train')
logger.enable_exception_hook()
writer = SummaryWriter()
# torch.autograd.set_detect_anomaly(True)


def train(args):
    # Initialize device
    logger.info("Training date time: {}".format(datetime.datetime.now()))
    logger.info("Training title: {}".format(args.title))
    logger.info("=========================")
    logger.info("Hyper arguments:")
    for arg_name, arg_value in vars(args).items():
        logger.info("{}: {}".format(arg_name, arg_value))
    logger.info("=========================")
    start_time = time.time()
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.device))
        torch.backends.cudnn.benchmark = True
        logger.info("Using device: CUDA_{}".format(args.device))
    else:
        logger.warn("Using device: CPU")


# Data loading and splitting
    logger.info("Start loading data.")

    source_dataset = Cell_dataset(root=os.path.join(
        Constants.DATA_FOLDER, Constants.DATASET[args.source_dataset]),
                                  type=args.source_dataset_type,
                                  transform=True,
                                  crop_size=args.patch_size,
                                  resize=args.image_resize,
                                  training_factor=args.training_scale_s)
    target_dataset = Cell_dataset(root=os.path.join(
        Constants.DATA_FOLDER, Constants.DATASET[args.target_dataset]),
                                  type=args.target_dataset_type,
                                  transform=False,
                                  crop_size=args.patch_size,
                                  resize=args.image_resize,
                                  training_factor=args.training_scale_t)
    target_dataset_train, target_dataset_valid = random_split(
        target_dataset, [args.training_ratio, 1 - args.training_ratio])
    source_dataset.train()
    target_dataset_train.dataset.train()
    target_dataset_valid.dataset.eval()
    source_dataloader = DataLoader(source_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   drop_last=True)
    target_dataloader_train = DataLoader(target_dataset_train,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         drop_last=True)
    target_dataloader_valid = DataLoader(target_dataset_valid,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         drop_last=True)

    logger.info(
        "Loading data completed. Elapsed time: {:.2f}sec.".format(time.time() -
                                                                  start_time))

    logger.info("Start initailizing model")
    if args.patch_size:
        insize = args.patch_size
    elif args.image_resize:
        insize = args.image_resize
    model = UDACounting(in_size=insize,
                        dropout=args.dropout,
                        momentum=args.BN_momentum)
    model.to(device)

    # Initialize the optimizer with weight decay and learning rate
    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=args.restart_step,
                                            T_mult=4,
                                            eta_min=1e-10)
    # scheduler = StepLR(optimizer, step_size=50, gamma=args.lr_decay)

    # Distinguish voxel-wise loss and count loss
    loss_voxel = torch.nn.MSELoss(reduction='mean')
    loss_domain = torch.nn.NLLLoss()
    loss_voxel = loss_voxel.to(device)
    loss_domain = loss_domain.to(device)
    count_mae = torch.nn.L1Loss(reduction='mean')

    logger.info("Initialization Completed. Elapsed time: {:.2f}sec".format(
        time.time() - start_time))

    train_time = time.time()
    count_MAE = AverageMeter()
    voxel_MSE = AverageMeter()
    for epoch in range(args.epoch):
        epoch_time = time.time()
        epoch_voxel_loss = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_dis_loss = AverageMeter()
        valid_voxel_loss = AverageMeter()
        valid_count_loss = AverageMeter()
        model.train()

        length = min(source_dataloader.__len__(),
                     target_dataloader_train.__len__())
        for batch_idx, (source_data, target_data) in enumerate(
                itertools.zip_longest(source_dataloader,
                                      target_dataloader_train)):
            if source_data is None or target_data is None:
                break
            p = float(batch_idx + epoch * length) / args.epoch / length
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_domain_label = torch.zeros(
                args.batch_size).long().to(device)
            target_domain_label = torch.ones(args.batch_size).long().to(device)

            img_s, dot_s, counts_s = source_data
            img_t, dot_t, counts_t = target_data

            img_s = img_s.to(device)
            dot_s = dot_s.to(device)
            img_t = img_t.to(device)
            dot_t = dot_t.to(device)

            model.source()
            with torch.set_grad_enabled(True):
                # Train on source domain
                output_s, domain_output_s = model(img_s, alpha)
                loss_s_voxel = loss_voxel(output_s, dot_s)
                if epoch < args.warm_start:
                    optimizer.zero_grad()
                    loss_s_voxel.backward()
                    optimizer.step()
                counts_pred_s = output_s.sum(
                    [1, 2, 3]).detach().cpu() / args.training_scale_s
                loss_s_count = count_mae(counts_pred_s, counts_s)
                loss_s_domain = loss_domain(domain_output_s,
                                            source_domain_label)

                # Separately align the positive and negative feature maps
                mask = output_s.detach()
                mask[torch.where(mask > 0)] = 1
                bg_s = torch.zeros_like(mask, device=device)
                output_t, domain_output_s_T = model(img_s * mask, alpha)
                output_f, domain_output_s_F = model(img_s * (1. - mask), alpha)
                loss_s_voxel_t = loss_voxel(output_t, dot_s)
                loss_s_voxel_f = loss_voxel(output_f, bg_s)
                loss_s_voxel_separated = loss_s_voxel_t + loss_s_voxel_f
                loss_s_domain_T = loss_domain(domain_output_s_T,
                                              source_domain_label)
                loss_s_domain_F = loss_domain(domain_output_s_F,
                                              source_domain_label)

                # Compute the loss of source domain
                # loss_s = (loss_s_voxel_separated) / (
                #     loss_s_domain_T + loss_s_domain_F + loss_s_domain)
                loss_s = 1 / loss_s_domain

                # Train on target domain
                model.target()
                # with torch.set_grad_enabled(True):
                output_t, domain_output_t = model(img_t, alpha)
                loss_t_voxel = loss_voxel(output_t, dot_t)
                counts_pred_t = output_t.sum(
                    [1, 2, 3]).detach().cpu() / args.training_scale_t
                loss_t_count = count_mae(counts_pred_t, counts_t)
                loss_t_domain = loss_domain(domain_output_t,
                                            target_domain_label)

                # Separately align the positive and negative feature maps
                mask = output_t.detach()
                bg_t = torch.zeros_like(mask, device=device)
                mask[torch.where(mask > 0)] = 1
                _, domain_output_t_T = model(img_t * mask, alpha)
                output_t_f, domain_output_t_F = model(img_t * (1. - mask),
                                                      alpha)
                loss_s_voxel_f = loss_voxel(output_t_f, bg_t)
                loss_t_domain_T = loss_domain(domain_output_t_T,
                                              target_domain_label)
                loss_t_domain_F = loss_domain(domain_output_t_F,
                                              target_domain_label)
                # loss_t = loss_s_voxel_f / (loss_t_domain_T + loss_t_domain_F +
                #                            loss_t_domain)
                loss_t = 1 / loss_t_domain
                if epoch < args.warm_start:
                    loss = loss_s_voxel
                else:
                    loss = loss_s + loss_t
                    loss = loss.neg()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Step the scheduler
            scheduler.step(epoch + batch_idx / length)

            # Update the loss meter
            epoch_voxel_loss.update(loss_s_voxel_separated.item())
            epoch_count_loss.update(loss_s_count.item())
            epoch_dis_loss.update(loss_s_domain_T.item() +
                                  loss_s_domain_F.item())
            epoch_voxel_loss.update(loss_t_voxel.item())
            epoch_count_loss.update(loss_t_count.item())
            epoch_dis_loss.update(loss_t_domain_T.item() +
                                  loss_t_domain_F.item())

            # Logging in Tensorboard
            if batch_idx % 100 == 0:
                writer.add_scalar(tag='Running loss of {}'.format(
                    Constants.LOG_NAME),
                                  scalar_value=loss.item(),
                                  global_step=epoch)

        # Validation
        model.eval()
        model.target()
        for batch_idx, target_data in enumerate(target_dataloader_valid):
            img_t, dot_t, counts_t = target_data
            with torch.no_grad():
                img_t = img_t.to(device)
                dot_t = dot_t.to(device)
                alpha = 0
                output, _ = model(img_t, alpha)
                loss = loss_voxel(output, dot_t)
                counts_pred = output.sum(
                    [1, 2, 3]).detach().cpu() / args.training_scale_t
                loss_count = count_mae(counts_pred, counts_t)

            # Update the loss meter
            valid_voxel_loss.update(loss.item())
            valid_count_loss.update(loss_count.item())

        # Tensorboard writer and logging out per epoch
        writer.add_scalars(main_tag='Epoch loss of {}'.format(
            Constants.LOG_NAME),
                           tag_scalar_dict={
                               'Train pixel loss': epoch_voxel_loss.get('avg'),
                               'Train MAE': epoch_count_loss.get('avg'),
                               'Distinguish MSE': epoch_dis_loss.get('avg'),
                               'Valid pixel loss': valid_voxel_loss.get('avg'),
                               'Valid MAE': valid_count_loss.get('avg')
                           },
                           global_step=epoch)
        logger.info("Epoch:{} Cost: {:.1f} sec\nTrain\
                    Loss: {:.2f}, \
                    Dis: {:.2f}, \
                    MAE: {:.2f}, \
                    \nValid\
                    Loss: {:.2f}, \
                    MAE: {:.2f}, \
                    ".format(epoch + 1,
                             time.time() - epoch_time,
                             epoch_voxel_loss.get('avg'),
                             epoch_dis_loss.get('avg'),
                             epoch_count_loss.get('avg'),
                             valid_voxel_loss.get('avg'),
                             valid_count_loss.get('avg')))
        voxel_MSE.update(valid_voxel_loss.get('avg'))
        count_MAE.update(valid_count_loss.get('avg'))
    logger.info("Training completed ({:.2f} sec)".format(time.time() -
                                                         train_time))
    logger.info("The best count MAE: {:.2f}".format(count_MAE.get('min')))
    logger.info("The best voxel loss: {:.2f}".format(voxel_MSE.get('min')))
    writer.close()
    model.source()
    model_stats_s = summary(model, input_data=[img_t, alpha], verbose=0)
    model.target()
    model_stats_t = summary(model, input_data=[img_t, alpha], verbose=0)
    logger.info("Source Model Summary:\n{}".format(str(model_stats_s)))
    logger.info("Target Model Summary:\n{}".format(str(model_stats_t)))
    logger.info("Finished.")

if __name__ == '__main__':
    parser = ArgParser()
    parser.load_arguments()
    args = parser.parse_args()
    train(args)
