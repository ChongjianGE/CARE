import os
import sys
import argparse
import time
import importlib
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.log import setup_logger
from utils import accuracy, adjust_learning_rate_iter, save_checkpoint, parse_devices, AvgMeter
from utils.torch_dist import reduce_tensor_sum
import ipdb


def cleanup():
    dist.destroy_process_group()


def main():
    pretrained_file_name = os.path.join(exp.output_dir, args.experiment_name)
    file_name = os.path.join(pretrained_file_name, "{}linear_eval_teacher".format(exp.save_folder_prefix))

    if args.local_rank == 0:
        if not os.path.exists(file_name):
            os.makedirs(file_name, exist_ok=True)

    logger = setup_logger(file_name, distributed_rank=args.local_rank, filename="eval_log.txt", mode="a")
    if args.local_rank == 0:
        logger.info("gpuid: {}, args: {}".format(args.local_rank, args))
    data_loader = exp.get_data_loader(batch_size=args.batch_size, is_distributed=args.nr_gpu > 1)

    train_loader, eval_loader = data_loader["train"], data_loader["eval"]
    model = exp.get_model().to(device)

    #  ------------------------------------------- load ckpt ------------------------------------ #
    ckpt_tar = os.path.join(pretrained_file_name, args.checkpoints + "_ckpt.pth.tar")
    map_location = 'cpu'
    ckpt = torch.load(ckpt_tar, map_location=map_location)

    state_dict = {k.replace("module.student_encoder.", ""): v for k, v in ckpt["model"].items()}
    state_dict_new = {k.replace("bot", "trans"): v for k, v in state_dict.items()}

    missing_keys = []
    matched_state_dict = {}
    if args.local_rank == 0:
        logger.info("here is the model state dict:{}".format(model.state_dict().keys()))

    for name, param in state_dict_new.items():
        if "encoder.{}".format(name) not in model.state_dict() or name.startswith("fc"):
            missing_keys.append(name)
        else:
            matched_state_dict["encoder.{}".format(name)] = param
    del state_dict_new
    del state_dict

    msg = model.load_state_dict(matched_state_dict, strict=False)
    del matched_state_dict

    # -------------------------------------- end of the tmp --------------------------------------- #
    if args.local_rank == 0:
        # logger.info(str(model))
        logger.info("Missing keys: {}".format(missing_keys))
        logger.info("Params {} are not loaded from matched state dict".format(msg.missing_keys))

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    optimizer = exp.get_optimizer_new(model.module, args.batch_size)
    world_size = torch.distributed.get_world_size() if args.distributed else 1
    cudnn.benchmark = True

    best_top1 = 0
    _best_top5 = 0
    best_top1_epoch = 0

    ITERS_PER_EPOCH = len(train_loader)
    args.lr = exp.basic_lr_per_img * args.batch_size
    args.total_epochs = exp.max_epochs
    args.scheduler = exp.scheduler
    args.milestones = exp.epoch_of_stage
    model.train()

    for epoch in range(0, args.total_epochs):
        if args.nr_gpu > 1:
            train_loader.sampler.set_epoch(epoch)

        for i, (inp, target) in enumerate(train_loader):
            data_time = time.time()
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            data_time = time.time() - data_time

            # forward
            logits, loss = model(inp, target)
            top1, top5 = accuracy(logits, target, (1, 5))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_count = epoch * ITERS_PER_EPOCH + i + 1
            lr = adjust_learning_rate_iter(optimizer, iter_count, args, ITERS_PER_EPOCH)

            if (i + 1) % exp.print_interval == 0 and args.local_rank == 0:
                logger.info(
                    "\tIter: [{}/{}], Epoch: [{}/{}], Data-Time: {:.3f}, LR: {:.4f},"
                    " Loss: {:.4f}, Top-1: {:.2f}, Top-5: {:.2f}".format(
                        i + 1, ITERS_PER_EPOCH, epoch + 1, args.total_epochs, data_time, lr, loss, top1, top5
                    )
                )

        if (epoch + 1) % exp.eval_interval == 0 or (epoch + 1) in [1, 2, 5, 7]:
            logger.info("start evaluation")
            model.eval()
            eval_top1, eval_top5 = run_eval(model, eval_loader)
            model.train()

            logger.info(
                "\tEval-Epoch: [{}/{}], Top-1: {:.2f}, Top-5: {:.2f}".format(
                    epoch + 1, args.total_epochs, eval_top1, eval_top5
                )
            )

            if eval_top1 > best_top1:
                is_best = True
                best_top1 = eval_top1
                _best_top5 = eval_top5
                best_top1_epoch = epoch + 1
            else:
                is_best = False

            logger.info(
                "\tBest Top-1 at epoch [{}/{}], Best Top-1: {:.2f}, Top-5: {:.2f}".format(
                    best_top1_epoch, args.total_epochs, best_top1, _best_top5
                )
            )
            if args.local_rank == 0:
                save_checkpoint(
                    {
                        "start_epoch": epoch + 1,
                        "classifier": model.state_dict(),
                        "best_top1": best_top1,
                        "_best_top5": _best_top5,
                        "best_top1_epoch": best_top1_epoch,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    file_name,
                    "linear_eval",
                )

    if args.local_rank == 0:
        print("Pre-training done.")
        print("Experiment name: {}".format(args.experiment_name))


def run_eval(model, eval_loader):
    top1 = AvgMeter()
    top5 = AvgMeter()

    with torch.no_grad():
        pbar = tqdm(range(len(eval_loader)))
        for _, (inp, target) in zip(pbar, eval_loader):
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            logits = model(inp)
            acc1, acc5 = accuracy(logits, target, (1, 5))
            acc1, acc5 = (
                reduce_tensor_sum(acc1) / dist.get_world_size(),
                reduce_tensor_sum(acc5) / dist.get_world_size(),
            )
            top1.update(acc1.item(), inp.size(0))
            top5.update(acc5.item(), inp.size(0))
    return top1.avg, top5.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CARE_LinearEvaluation")
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("-expn", "--experiment_name", type=str, default="baseline-")
    parser.add_argument("-ckpt", "--checkpoints", type=str, default="last_epoch")
    parser.add_argument("--log_path", default="/apdcephfs/share_1290939/chongjiange/code/unsup_momentum/train_new_log/", type=str, help="the path of the logging file")

    # distributed
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-d", "--devices", default="0-7", type=str, help="device for training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    args.nr_gpu = torch.cuda.device_count()
    if args.local_rank == 0:
        print("V1 Using", torch.cuda.device_count(), "GPUs per node!")

    if not args.exp_file:
        from care.exps.arxiv.linear_eval_exp import Exp
        exp = Exp()
    else:
        import importlib
        sys.path.insert(0, os.path.dirname(args.exp_file))
        current_exp = importlib.import_module(os.path.basename(args.exp_file).split(".")[0])
        exp = current_exp.Exp(args)

    print("The print interval is {}".format(exp.print_interval))

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
