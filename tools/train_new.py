import os
import sys
import argparse
import time
import random
import warnings
import subprocess
import importlib
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.log import setup_logger
from utils import adjust_learning_rate_iter, save_checkpoint, parse_devices, AvgMeter
from utils.torch_dist import configure_nccl, synchronize
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast as autocast, GradScaler


def cleanup():
    dist.destroy_process_group()


def main():
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    if args.local_rank == 0:
        if not os.path.exists(file_name):
            os.makedirs(file_name, exist_ok=True)
        writer = SummaryWriter(os.path.join(file_name, 'runs'))

    logger = setup_logger(file_name, distributed_rank=args.local_rank, filename="train_log.txt", mode="a")
    logger.info("gpuid: {}, args: {}".format(args.local_rank, args))

    train_loader = exp.get_data_loader(batch_size=args.batchsize, is_distributed=args.nr_gpu > 1, if_transformer=False)["train"]
    model = exp.get_model().to(device)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    optimizer = exp.get_optimizer_new(model.module, args.batchsize)

    world_size = torch.distributed.get_world_size() if args.distributed else 1

    cudnn.benchmark = True

    # -----------------------------start training-----------------------------#
    model.train()
    ITERS_PER_EPOCH = len(train_loader)
    if args.local_rank == 0:
        logger.info("Training start...")
        logger.info("Here is the logging file"+str(file_name))
        # logger.info(str(model))

    args.lr = exp.basic_lr_per_img * args.batchsize
    args.warmup_epochs = exp.warmup_epochs
    args.total_epochs = exp.max_epoch
    iter_count = 0

    scaler = GradScaler()
    for epoch in range(0, args.total_epochs):
        if args.nr_gpu > 1:
            train_loader.sampler.set_epoch(epoch)
        batch_time_meter = AvgMeter()

        for i, (inps, target) in enumerate(train_loader):
            iter_count += 1
            iter_start_time = time.time()

            for indx in range(len(inps)):
                inps[indx] = inps[indx].to(device, non_blocking=True)

            data_time = time.time() - iter_start_time
            with autocast():
                loss, con_l, con2_l, fea_l = model(inps, update_param=True)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr = adjust_learning_rate_iter(optimizer, iter_count, args, ITERS_PER_EPOCH)
            batch_time_meter.update(time.time() - iter_start_time)

            # log_interval = 1
            log_interval = exp.print_interval
            if args.local_rank == 0 and (i + 1) % log_interval == 0:
                remain_time = (ITERS_PER_EPOCH * exp.max_epoch - iter_count) * batch_time_meter.avg
                t_m, t_s = divmod(remain_time, 60)
                t_h, t_m = divmod(t_m, 60)
                t_d, t_h = divmod(t_h, 24)
                remain_time = "{}d.{:02d}h.{:02d}m".format(int(t_d), int(t_h), int(t_m))

                logger.info(
                    "[{}/{}], remain:{}, It:[{}/{}], Data-Time:{:.3f}, LR:{:.4f}, Loss:{:.2f}, CON_Loss:{:.2f}, "
                    "CON2_Loss:{:.2f}, FEA_Loss:{:.4f}".format(
                        epoch + 1, args.total_epochs, remain_time, i + 1, ITERS_PER_EPOCH, data_time, lr, loss, con_l, con2_l, fea_l
                    )
                )

            log_tensorboard_interval = 50
            if args.local_rank == 0 and (i + 1) % log_tensorboard_interval == 0:
                writer.add_scalar('con_l', con_l, iter_count)
                writer.add_scalar('con2_l', con2_l, iter_count)
                writer.add_scalar('fea_l', fea_l, iter_count)
                writer.add_scalar('loss', loss, iter_count)
                writer.add_scalar('lr', lr, iter_count)

        if args.local_rank == 0:
            logger.info(
                "Train-Epoch: [{}/{}], LR: {:.4f}, Con-Loss: {:.2f}".format(epoch + 1, args.total_epochs, lr, loss)
            )

            save_checkpoint(
                {"start_epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                False,
                file_name,
                "last_epoch",
            )

            if epoch in [9, 19, 29, 39, 49, 74, 99, 149, 199, 249, 299, 349, 399, 449, 499, 549, 599, 649, 699, 749, 799]:
                save_checkpoint(
                    {"start_epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    False,
                    file_name,
                    str(epoch+1),
                )

    if args.local_rank == 0:
        print("Pre-training of experiment: {} is done.".format(args.exp_file))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CARE")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)

    # optimization
    parser.add_argument(
        "--scheduler",
        type=str,
        default="warmcos",
        choices=["warmcos", "cos", "linear", "multistep", "step"],
        help="type of scheduler",
    )

    # distributed
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("-n", "--n_views", type=int, default=2)
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batchsize", type=int, default=256, help="batch size")
    parser.add_argument("-d", "--devices", default="0-7", type=str, help="device for training")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("--log_path", default="/apdcephfs/share_1290939/chongjiange/code/unsup_momentum/train_new_log/", type=str, help="the path of the logging file")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    args.nr_gpu = torch.cuda.device_count()
    if args.local_rank == 0:
        print("V1 Using", torch.cuda.device_count(), "GPUs per node!")

    if not args.exp_file:
        from care.exps.arxiv import care_exp
        exp = care_exp.Exp(args)
    else:
        sys.path.insert(0, os.path.dirname(args.exp_file))
        current_exp = importlib.import_module(os.path.basename(args.exp_file).split(".")[0])
        exp = current_exp.Exp(args)

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        if args.local_rank == 0:
            if os.path.exists("./" + args.experiment_name + "ip_add.txt"):
                os.remove("./" + args.experiment_name + "ip_add.txt")

    print("Rank {} initialization finished.".format(args.local_rank))
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    main()

    if args.distributed:
        cleanup()
