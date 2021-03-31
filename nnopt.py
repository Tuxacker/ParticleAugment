import argparse
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as torchmp
import torch.cuda.amp as amp
#import torch.optim
#import torch.utils.data
#import torch.utils.data.distributed
#import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#import torchvision.models as models

import numpy as np

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
except ImportError:
    raise ImportError("Please install all requirements, NVIDIA DALI seems to be missing!")

from components.model_factory import model_from_config
from components.loss_factory import loss_from_config
from components.optimizer_factory import optimizer_from_config
from components.lrs_factory import lrs_from_config
from data.dali_ra import get_lcra_train_iterator, get_lcra_val_iterator
from utils.config_src import get_global_config

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--local_rank", nargs='?', const=-1, default=-1, type=int)
    parser.add_argument("--resume", nargs='?', default="")
    args = parser.parse_args()
    return args

# def parse():
#     model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))

#     parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#     parser.add_argument('data', metavar='DIR', nargs='*',
#                         help='path(s) to dataset (if one path is provided, it is assumed\n' +
#                        'to have subdirectories named "train" and "val"; alternatively,\n' +
#                        'train and val paths can be specified directly by providing both paths as arguments)')
#     parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
#                         choices=model_names,
#                         help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
#     parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                         help='number of data loading workers (default: 4)')
#     parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                         help='number of total epochs to run')
#     parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                         help='manual epoch number (useful on restarts)')
#     parser.add_argument('-b', '--batch-size', default=256, type=int,
#                         metavar='N', help='mini-batch size per process (default: 256)')
#     parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                         metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
#     parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                         help='momentum')
#     parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                         metavar='W', help='weight decay (default: 1e-4)')
#     parser.add_argument('--print-freq', '-p', default=10, type=int,
#                         metavar='N', help='print frequency (default: 10)')
#     parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                         help='path to latest checkpoint (default: none)')
#     parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                         help='evaluate model on validation set')
#     parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                         help='use pre-trained model')

#     parser.add_argument('--dali_cpu', action='store_true',
#                         help='Runs CPU based version of DALI pipeline.')
#     parser.add_argument('--prof', default=-1, type=int,
#                         help='Only run 10 iterations for profiling.')
#     parser.add_argument('--deterministic', action='store_true')

#     parser.add_argument("--local_rank", default=0, type=int)
#     parser.add_argument('--sync_bn', action='store_true',
#                         help='enabling apex sync BN.')

#     parser.add_argument('--opt-level', type=str, default=None)
#     parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
#     parser.add_argument('--loss-scale', type=str, default=None)
#     parser.add_argument('--channels-last', type=bool, default=False)
#     parser.add_argument('-t', '--test', action='store_true',
#                         help='Launch test mode with preset arguments')
#     args = parser.parse_args()
#     return args


def main(local_rank=-1, world_size=1):

    global best_prec1, _args, config

    best_prec1 = 0.0

    #_args = parse()

    config = get_global_config()

    if local_rank > -1:
        config.distributed = True
        config.local_rank = local_rank
        if local_rank > 0:
            f = open("/dev/null", "w")
            sys.stdout = f
    else:
        config.local_rank = 0

    if not hasattr(config, "distributed"):
        config.distributed = False

    if 'WORLD_SIZE' in os.environ:
        config.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        os.environ['WORLD_SIZE'] = str(world_size)

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = "127.0.0.1"

    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = "2222"

    print("CUDNN VERSION: {}".format(cudnn.version()))

    cudnn.benchmark = True
    cudnn.deterministic = False

    if config.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(config.local_rank)
        torch.set_printoptions(precision=10)

    if not hasattr(config, "device_id"):
        config.device_id = 0
        config.world_size = 1

    if config.distributed:
        config.device_id = config.local_rank
        os.environ['RANK'] = str(config.local_rank)
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(config.local_rank)
        torch.cuda.set_device(config.device_id)
        dist.init_process_group(backend='nccl', init_method='env://')
        config.world_size = dist.get_world_size()

    config.total_batch_size = config.world_size * config.batch_size
    assert cudnn.enabled, "CUDNN backend needs to be enabled."

    model = model_from_config(config)

    if config.sync_bn:
        print("Converted model BN to SyncBatchNorm")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(config.device_id)

    # Scale learning rate based on global batch size
    config.lr = config.lr * float(config.batch_size * config.world_size) / 256.0

    config.start_epoch = 0

    if config.distributed:
        model = DDP(model, device_ids=[config.device_id], output_device=config.device_id)

    criterion = loss_from_config(config)

    optimizer = optimizer_from_config(config, model)

    lr_scheduler = lrs_from_config(config, optimizer)

    scaler = amp.GradScaler() if config.amp else None

    # Optionally resume from a checkpoint
    if config.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(config.resume):
                print("=> loading checkpoint '{}'".format(config.resume))
                checkpoint = torch.load(config.resume, map_location = lambda storage, loc: storage.cuda(config.device_id))
                config.start_epoch = checkpoint['epoch']
                global best_prec1
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                if checkpoint['scaler'] is not None:
                    scaler.load_state_dict(checkpoint['scaler'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(config.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(config.resume))
        if config.distributed:
            dist.barrier()
        resume()

    trainp, train_iterator = get_lcra_train_iterator(config.train_dataset_path, config.batch_size, 16, config.device_id, config.world_size, config.ra_m, last_batch_policy=LastBatchPolicy.PARTIAL, last_batch_padded=True)
    trainp.build()
    train_loader = DALIClassificationIterator(trainp, last_batch_policy=LastBatchPolicy.PARTIAL, dynamic_shape=True)

    valp, val_iterator = get_lcra_val_iterator(config.val_dataset_path, config.batch_size, 16, config.device_id, config.world_size)
    valp.build()
    val_loader = DALIClassificationIterator(valp, last_batch_policy=LastBatchPolicy.PARTIAL, dynamic_shape=True)

    if config.evaluate:
        validate(val_loader, val_loader, model, criterion)
        return

    total_time = AverageMeter()
    for epoch in range(config.start_epoch, config.epochs):
        # train for one epoch
        avg_train_time = train(train_loader, train_iterator, model, criterion, optimizer, lr_scheduler, epoch, scaler)
        total_time.update(avg_train_time)
        if config.test:
            break

        if epoch % config.validation_interval == 0:
            # evaluate on validation set
            [prec1, prec5] = validate(val_loader, val_iterator, model, criterion)

            # remember best prec@1 and save checkpoint
            if config.local_rank == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'scaler': scaler.state_dict()
                }, is_best)
                if epoch == config.epochs - 1:
                    print('##Top-1 {0}\n'
                        '##Top-5 {1}\n'
                        '##Perf  {2}'.format(
                        prec1,
                        prec5,
                        config.total_batch_size / total_time.avg))
            val_loader.reset()
        train_loader.reset()

def train(train_loader, train_iterator, model, criterion, optimizer, lr_scheduler, epoch, scaler=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    epoch_start_time = time.time()

    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        train_loader_len = int(np.ceil(len(train_iterator) / config.batch_size))

        if config.profile >= 0 and i == config.profile:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if config.profile >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        #adjust_learning_rate(optimizer, epoch, i, train_loader_len)
        if config.test:
            if i > 10:
                break

        with amp.autocast(enabled=config.amp):
            # compute output
            if config.profile >= 0: torch.cuda.nvtx.range_push("forward")
            output = model(input)
            if config.profile >= 0: torch.cuda.nvtx.range_pop()
            loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if config.profile >= 0: torch.cuda.nvtx.range_push("backward")
        if config.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
             loss.backward()
        if config.profile >= 0: torch.cuda.nvtx.range_pop()

        if config.profile >= 0: torch.cuda.nvtx.range_push("optimizer.step() and lr_scheduler.step()")
        optimizer.step()
        lr_scheduler.step()
        if config.profile >= 0: torch.cuda.nvtx.range_pop()

        if i % config.debug_print_freq == 0 and i > 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            #pylint: disable=unbalanced-tuple-unpacking
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if config.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(reduced_loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / config.debug_print_freq)
            end = time.time()

            if config.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, train_loader_len,
                       config.world_size*config.batch_size/batch_time.val,
                       config.world_size*config.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))

        # Pop range "Body of iteration {}".format(i)
        if config.profile >= 0: torch.cuda.nvtx.range_pop()

        if config.profile >= 0 and i == config.profile + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()

    print('Epoch: [{}]\t\tElapsed {:.3f}'.format(epoch, time.time() - epoch_start_time))

    return batch_time.avg

def validate(val_loader, val_iterator, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        val_loader_len = int(np.ceil(len(val_iterator) / config.batch_size))

        # compute output
        with torch.no_grad(), amp.autocast(enabled=config.amp):
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        #pylint: disable=unbalanced-tuple-unpacking
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if config.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if config.local_rank == 0 and i % config.debug_print_freq == 0 and i > 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len,
                   config.world_size * config.batch_size / batch_time.val,
                   config.world_size * config.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print('Validation:\t\tPrec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = config.lr*(0.1**factor)

    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= config.world_size
    return rt

if __name__ == '__main__':

    if len(sys.argv) > 1:
        torchmp.start_processes(main, nprocs=int(sys.argv[1]), args=(2,), start_method="forkserver")
    else:
        main()