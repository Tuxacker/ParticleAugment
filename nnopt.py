import argparse
import glob
#from contextlib import ContextDecorator
import os
import shutil
import sys
import time
import logging
import datetime
from pathlib import Path

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
import json

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator, LastBatchPolicy
except ImportError:
    raise ImportError("Please install all requirements, NVIDIA DALI seems to be missing!")

from components.model_factory import model_from_config
from components.loss_factory import loss_from_config
from components.optimizer_factory import optimizer_from_config
from components.lrs_factory import lrs_from_config
from data.dali_ra import get_lcra_train_iterator, get_lcra_val_iterator
from data.pim_loader import get_cifar_10_train_loader, get_cifar_10_val_loader
from data.pim_ra import RandAugment
from utils.config_src import get_global_config


def main(local_rank=-1, world_size=1, overrides=None):
    start_time = time.time()

    global best_prec1, _args, config, logger

    logFormatter = logging.Formatter("%(asctime)s %(message)s")
    logger = logging.getLogger(str(local_rank))
    logger.setLevel(logging.INFO)
    # Fix for dumb duplication bug
    logger.propagate = False

    suffix = "" if local_rank == -1 else "_r" + str(local_rank)

    file_name = datetime.datetime.now().strftime("%y%m%d%H%M%S") + suffix + ".log"
    #work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work_dir")
    #Path(work_dir).mkdir(parents=True, exist_ok=True)
    log_path = file_name # os.path.join(work_dir, file_name) 

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    if local_rank < 1:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

    best_prec1 = 0.0

    #_args = parse()

    config = get_global_config()

    if type(overrides) is dict:
        config = merge_dict(config, overrides)

    if local_rank > -1:
        config.distributed = True
        config.local_rank = local_rank
        if local_rank > 0:
            #f = open("/dev/null", "w")
            #sys.stdout = f
            pass
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

    logger.info("CUDNN VERSION: {}".format(cudnn.version()))

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

    logger.info("Using config:")
    logger.info(json.dumps(config, indent=4))

    model = model_from_config(config)

    if config.sync_bn:
        logger.info("Converted model BN to SyncBatchNorm")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(config.device_id)

    # Scale learning rate based on global batch size
    #config.lr = config.lr * float(config.batch_size * config.world_size) / 256.0
    config.lr *= config.world_size

    config.start_epoch = 0

    if config.distributed:
        logger.info("Using DDP")
        model = DDP(model, device_ids=[config.device_id], output_device=config.device_id, find_unused_parameters=config.model=="shake_shake")

    criterion = loss_from_config(config).to(config.device_id)

    optimizer = optimizer_from_config(config, model)

    lr_scheduler = lrs_from_config(config, optimizer)

    scaler = amp.GradScaler() if config.amp else None

    # Optionally resume from a checkpoint
    if config.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(config.resume):
                logger.info("=> loading checkpoint '{}'".format(config.resume))
                checkpoint = torch.load(config.resume, map_location = lambda storage, loc: storage.cuda(config.device_id))
                config.start_epoch = checkpoint['epoch']
                global best_prec1
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                if checkpoint['scaler'] is not None:
                    scaler.load_state_dict(checkpoint['scaler'])
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                      .format(config.resume, checkpoint['epoch']))
            else:
                logger.info("=> no checkpoint found at '{}'".format(config.resume))
        if config.distributed:
            dist.barrier()
        resume()

    # TODO: Maybe change DALI iterator reset trigger to config setting
    if config.dataloader == "dali":
        trainp, train_iterator = get_lcra_train_iterator(config.train_dataset_path, config.batch_size, config.data_threads, config.device_id, config.world_size, config.ra_m, last_batch_policy=LastBatchPolicy.PARTIAL, last_batch_padded=True)
        trainp.build()
        train_loader = DALIClassificationIterator(trainp, last_batch_policy=LastBatchPolicy.PARTIAL, dynamic_shape=True)

        if config.filter.extra_train:
            filtp, filtrain_iterator = get_lcra_train_iterator(config.train_dataset_path, config.batch_size, config.data_threads, config.device_id, config.world_size, config.ra_m, last_batch_policy=LastBatchPolicy.PARTIAL, last_batch_padded=True, sublist=list(range(config.batch_size*200)))
            filtp.build()
            filtrain_loader = DALIClassificationIterator(filtp, last_batch_policy=LastBatchPolicy.PARTIAL, dynamic_shape=True)

        valp, val_iterator = get_lcra_val_iterator(config.val_dataset_path, config.batch_size, config.data_threads, config.device_id, config.world_size)
        valp.build()
        val_loader = DALIClassificationIterator(valp, last_batch_policy=LastBatchPolicy.PARTIAL, dynamic_shape=True)

        filp, fil_iterator = get_lcra_train_iterator(config.train_dataset_path, config.batch_size, config.data_threads, config.device_id, config.world_size, config.ra_m, last_batch_policy=LastBatchPolicy.PARTIAL, last_batch_padded=True, shuffle=False, sublist=list(range(config.batch_size*4)))
        filp.build()
        fil_loader = DALIClassificationIterator(filp, last_batch_policy=LastBatchPolicy.PARTIAL, dynamic_shape=True, prepare_first_batch=False)
        #fil_iterator.name = "fil_iterator
    elif config.dataloader == "pim":
        train_loader, train_iterator = get_cifar_10_train_loader(config, config.train_dataset_path, config.batch_size, config.data_threads, config.device_id, config.world_size)
        if config.filter.extra_train:
            filtrain_loader, filtrain_iterator = get_cifar_10_train_loader(config, config.train_dataset_path, config.batch_size, config.data_threads_pft, config.device_id, config.world_size, subset=list(range(config.batch_size*200)))
        val_loader = get_cifar_10_val_loader(config, config.val_dataset_path, config.batch_size, config.data_threads, config.device_id, config.world_size)
        val_iterator = None
        indices = train_loader.dataset.get_balanced_subset(config.batch_size*4)
        fil_loader, fil_iterator = get_cifar_10_train_loader(config, config.train_dataset_path, config.batch_size, 0, config.device_id, config.world_size, subset=indices)


    if config.evaluate:
        validate(val_loader, val_iterator, model, criterion)
        return

    p_filter = ParticleFilter(model, train_iterator, fil_loader, fil_iterator, criterion)

    total_time = AverageMeter()

    logger.info("Initial LR: {}".format(optimizer.param_groups[0]["lr"]))

    epoch = config.start_epoch

    prec1_hist = []
    saved_val_epochs = []

    #move_old()

    def reload_checkpoint(filename, prefix="default"):
        checkpoint = torch.load(filename, map_location = lambda storage, loc: storage.cuda(config.device_id))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        p_filter.load_state_dict(checkpoint['pf'])
        prec1_hist = checkpoint['prec1_hist']
        saved_val_epochs = checkpoint['saved_val_epochs']
        if checkpoint['scaler'] is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info("[{}]=> loaded checkpoint '{}' (epoch {})"
                .format(prefix, filename, checkpoint['epoch'] + 1))
        return checkpoint['epoch'], prec1_hist, saved_val_epochs

    retry_epoch = -1
    num_retries = 0

    while epoch < config.epochs:
        # train for one epoch
        if config.dataloader == "dali":
            train_iterator.set_epoch(epoch)
        p_filter.epoch = epoch
        if not config.filter.extra_train:
            p_filter.p_enter(epoch, config.filter.enabled)

        if config.distributed:
            dist.barrier()

        avg_train_time = train(train_loader, train_iterator, model, criterion, optimizer, lr_scheduler, epoch, scaler)

        if config.distributed:
            dist.barrier()

        if config.local_rank == 0:

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'pf': p_filter.state_dict(),
                'prec1_hist': prec1_hist,
                'saved_val_epochs': saved_val_epochs
            })

            retain_latest(to_keep=config.keep_epochs)

        if config.distributed:
            dist.barrier()

        if not config.filter.extra_train:
            p_filter.p_exit(epoch, config.filter.enabled)
        elif epoch > config.start_epoch and config.filter.enabled and epoch % config.filter.interval == 0:
            p_filter.p_enter(epoch, config.filter.enabled)
            avg_train_time = train(filtrain_loader, filtrain_iterator, model, criterion, optimizer, lr_scheduler, epoch, scaler)
            p_filter.p_exit(epoch, config.filter.enabled)
            if config.filter.enabled and epoch % config.filter.interval == 0:
                reload_checkpoint(get_latest(), "pf")
            if type(filtrain_loader) == DALIClassificationIterator or type(filtrain_loader) == DALIGenericIterator:
                filtrain_loader.reset()

        total_time.update(avg_train_time)
        if config.test:
            break

        if config.distributed:
            dist.barrier()

        if epoch % config.validation_interval == 0 or epoch == config.epochs - 1:
            # evaluate on validation set
            [prec1, prec5, _] = validate(val_loader, val_iterator, model, criterion)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            prec1_hist.append(prec1)
            saved_val_epochs.append(epoch)
            best_prec1 = max(prec1, best_prec1)
            if config.local_rank == 0:
                copy_best(is_best)
            if epoch == config.epochs - 1:
                logger.info('##Top-1 {0}\n'
                    '##Top-5 {1}\n'
                    '##Perf  {2}'.format(
                    best_prec1,
                    prec5,
                    config.total_batch_size / total_time.avg))
                elapsed = time.time() - start_time
                logger.info("Total runtime: {:.2f}s (GPU seconds: {:.2f}s, {} GPUs)".format(elapsed, elapsed * world_size, world_size))
            if type(val_loader) == DALIClassificationIterator or type(val_loader) == DALIGenericIterator:
                val_loader.reset()
        if type(train_loader) == DALIClassificationIterator or type(train_loader) == DALIGenericIterator:
            train_loader.reset()

        if config.distributed:
            dist.barrier()

        if config.local_rank == 0:

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'pf': p_filter.state_dict(),
                'prec1_hist': prec1_hist,
                'saved_val_epochs': saved_val_epochs
            })

        if config.distributed:
            dist.barrier()

        if (epoch % config.validation_interval == 0 or epoch == config.epochs - 1) and len(prec1_hist) > 1 and config.retrain_on_val_loss:
            if prec1_hist[-1] - prec1_hist[-2] < - config.retrain_acc_threshold * min(1, num_retries):
                if epoch != retry_epoch:
                    retry_epoch = epoch
                    num_retries = 1
                else:
                    num_retries += 1
                if num_retries > config.num_retries and not config.reset_filter_on_val_loss:
                    retry_epoch = -1
                    num_retries = 0
                else:
                    logger.info("Reloading epoch {}".format(saved_val_epochs[-2] + 1))
                    #logger.info(prec1_hist)
                    #logger.info(saved_val_epochs)
                    epoch, prec1_hist, saved_val_epochs = reload_checkpoint("checkpoint-e{:04d}.pth.tar".format(saved_val_epochs[-2] + 1), "acc_drop")
                    if config.local_rank == 0:
                        checkpoints = sorted(glob.glob("checkpoint-e*.pth.tar"))
                        #logger.info(prec1_hist)
                        #logger.info(saved_val_epochs)
                        #logger.info(checkpoints)
                        #logger.info(checkpoints.index("checkpoint-e{:04d}.pth.tar".format(saved_val_epochs[-1] + 1))+1)
                        for file in checkpoints[checkpoints.index("checkpoint-e{:04d}.pth.tar".format(saved_val_epochs[-1] + 1))+1:]:
                            if os.path.exists(file):
                                logger.info("[acc drop] Removing {}".format(file))
                                os.remove(file)

                    if num_retries > config.num_retries:
                        p_filter.init_particles()
                        num_retries = 0
            
                    if config.distributed:
                        dist.barrier()

        epoch += 1

def train(train_loader, train_iterator, model, criterion, optimizer, lr_scheduler, epoch, scaler=None):
    #import cProfile, pstats
    #pr = cProfile.Profile()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    epoch_start_time = time.time()

    #pr.enable()
    for i, data in enumerate(train_loader):
        #pr.disable()
        if config.dataloader == "dali":
            input = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
            train_loader_len = int(np.ceil(len(train_iterator) / config.batch_size)) 
        else:
            input = data[0].to(config.device_id)
            target = data[1].to(config.device_id)
            train_loader_len = len(train_loader)


        if config.profile >= 0 and i == config.profile:
            logger.info("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if config.profile >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        #adjust_learning_rate(optimizer, epoch, i, train_loader_len)
        if config.test:
            if i > 10:
                break

        optimizer.zero_grad()

        with amp.autocast(enabled=config.amp):
            # compute output
            if config.profile >= 0: torch.cuda.nvtx.range_push("forward")
            output = model(input)
            if config.profile >= 0: torch.cuda.nvtx.range_pop()
            loss = criterion(output, target)

            if config.profile >= 0: torch.cuda.nvtx.range_push("backward and optimizer")
            if config.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
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
                logger.info('Train: E[{0:03d}/{me:03d}] B[{1:04d}/{2:04d}]'
                      ' Time:{batch_time.val:.3f} ({batch_time.avg:.3f})'
                      ' Speed:{3:.3f} ({4:.3f})'
                      ' Loss:{loss.val:6.4f} ({loss.avg:6.4f})'
                      ' Acc@1:{top1.val:6.3f} ({top1.avg:6.3f})'
                      ' Acc@5:{top5.val:6.3f} ({top5.avg:6.3f})'.format(
                       epoch + 1, i, train_loader_len,
                       config.world_size*config.batch_size/batch_time.val,
                       config.world_size*config.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5, me=config.epochs))

        # Pop range "Body of iteration {}".format(i)
        if config.profile >= 0: torch.cuda.nvtx.range_pop()

        if config.profile >= 0 and i == config.profile + 10:
            logger.info("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()
        #pr.enable()
    #pr.disable()

    logger.info('Epoch: [{}]\t\tElapsed {:.3f}, LR:{}'.format(epoch + 1, time.time() - epoch_start_time, optimizer.param_groups[0]["lr"]))

    #if config.dataloader == "pim":
    #    logger.info("RA apply stats: {}".format(train_iterator.get_op_statistics()))
    #    logger.info("RA count stats: {}".format(train_iterator.get_op_statistics(called=True)))

    lr_scheduler.step()

    #pr.print_stats(sort=pstats.SortKey.CUMULATIVE)

    return batch_time.avg

def validate(val_loader, val_iterator, model, criterion, suppress_output=False, calc_acc=True):
    #import cProfile, pstats
    #pr = cProfile.Profile()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    #pr.enable()
    for i, data in enumerate(val_loader):
        #pr.disable()
        if config.dataloader == "dali":
            input = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
            val_loader_len = int(np.ceil(len(val_iterator) / config.batch_size)) 
        else:
            input = data[0].to(config.device_id)
            target = data[1].to(config.device_id)
            val_loader_len = len(val_loader)

        # compute output
        with torch.no_grad(), amp.autocast(enabled=config.amp):
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            #pylint: disable=unbalanced-tuple-unpacking
            if calc_acc:
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            if config.distributed:
                reduced_loss = reduce_tensor(loss.data)
                if calc_acc:
                    prec1 = reduce_tensor(prec1)
                    prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            losses.update(reduced_loss.item(), input.size(0))
            if calc_acc:
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if config.local_rank == 0 and i % config.debug_print_freq == 0 and i > 0:
            logger.info('  Val: B[{0:04d}/{1:04d}]'
                  ' Time:{batch_time.val:.3f} ({batch_time.avg:.3f})'
                  ' Speed:{2:.3f} ({3:.3f})'
                  ' Loss:{loss.val:6.4f} ({loss.avg:6.4f})'
                  ' Acc@1:{top1.val:6.3f} ({top1.avg:6.3f})'
                  ' Acc@5:{top5.val:6.3f} ({top5.avg:6.3f})'.format(
                   i, val_loader_len,
                   config.world_size * config.batch_size / batch_time.val,
                   config.world_size * config.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
        #pr.enable()
    #pr.disable()

    if not suppress_output:
        logger.info("  Val: Loss:{loss.avg:6.4f} Acc@1: {top1.avg:5.3f}, Acc@5: {top5.avg:5.3f}"
            .format(loss=losses, top1=top1, top5=top5))

    #pr.print_stats(sort=pstats.SortKey.CUMULATIVE)

    return [top1.avg, top5.avg, losses.val]

def retain_latest(filename='checkpoint*.pth.tar', to_keep=10):
    if to_keep < config.validation_interval:
        raise ValueError("Not keeping enough epochs: {} < {}".format(to_keep, config.validation_interval))
    file_list = sorted(glob.glob(filename))
    if len(file_list) > to_keep:
        for file in file_list[:-5]:
            os.remove(file)

def get_latest(filename='checkpoint*.pth.tar'):
    file_list = sorted(glob.glob(filename))
    return file_list[-1]

def move_old():
    file_list = sorted(glob.glob("checkpoint*.pth.tar"))
    best_exists = os.path.exists("model_best.pth.tar")
    if len(file_list) > 0 or best_exists:
        folder_name = datetime.datetime.now().strftime("%y%m%d%H%M%S") + "-backup"
        os.mkdir(folder_name)
        if len(file_list) > 0:
            logger.info("Moving old checkpoints to {}".format(folder_name))
            for file in file_list:
                shutil.move(file, os.path.join(folder_name, file))
        if best_exists:
            logger.info("Moving old best checkpoint to {}".format(folder_name))
            file = "model_best.pth.tar"
            shutil.move(file, os.path.join(folder_name, file))

def save_checkpoint(state, filename='checkpoint-e{:04d}.pth.tar'):
    torch.save(state, filename.format(state["epoch"] + 1))

def copy_best(is_best, filename='checkpoint*.pth.tar'):
    file_list = sorted(glob.glob(filename))
    filename_newest = file_list[-1]
    if is_best:
        logger.info("Copying {} to model_best.pth.tar".format(filename_newest))
        shutil.copyfile(filename_newest, 'model_best.pth.tar')


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


class ParticleFilter: #(ContextDecorator):

    def __init__(self, model, train_iter, test_loader, test_iter, loss, seed=None):
        self.model = model
        self.train_iter = train_iter
        self.test_loader = test_loader
        self.test_iter = test_iter
        self.loss = loss
        self.n_states = 12 if config.dataloader == "dali" else 15
        self.particles = np.zeros((config.filter.n_particles, self.n_states), dtype=np.float32)
        self.weights = np.ones(config.filter.n_particles, dtype=np.float32) / config.filter.n_particles
        self.rng = np.random.default_rng(seed)
        self.init_particles()
        self.old_losses = np.zeros((config.filter.n_particles + 1), dtype=np.float32)
        self.new_losses = np.zeros((config.filter.n_particles + 1), dtype=np.float32)
        self.epoch = 0

    def state_dict(self):
        return {"particles": self.particles, "weights": self.weights, "n_states": self.n_states}

    def load_state_dict(self, state_dict):
        self.particles = state_dict["particles"]
        self.weights = state_dict["weights"]
        self.n_states = state_dict["n_states"]

    def init_particles(self):
        self.weights = np.ones(config.filter.n_particles, dtype=np.float32) / config.filter.n_particles
        for i in range(config.filter.n_particles):
            indices = self.rng.choice(self.n_states, config.ra_n, replace=False)
            for index in indices:
                self.particles[i, index] = 0.5

    def add_noise(self):
         self.particles += self.rng.normal(0.0, config.filter.std, self.particles.shape)
         self.particles = np.clip(self.particles, 0.0, 1.0)

    def nomalize_weights(self):
        self.weights = self.weights / np.sum(self.weights)
        self.weights[-1] = 1 - np.sum(self.weights[:-1])
        self.weights = np.clip(self.weights, 0.0, 1.0)

    def resample(self):
        p_copy = np.zeros_like(self.particles)
        if config.filter.only_good:
            good_policy_ids = self.weights * self.weights.shape[0] > config.filter.good_threshold
            particles = np.array([p for i, p in enumerate(self.particles) if good_policy_ids[i]])
            weights = self.weights[good_policy_ids]
            weights = weights / np.sum(weights)
            weights[-1] = 1 - np.sum(weights[:-1])
            weights = np.clip(weights, 0.0, 1.0)
        else:
            particles = self.particles
            weights = self.weights
        for i in range(config.filter.n_particles):
            index = self.rng.choice(len(weights), 1, p=weights)
            p_copy[i] = particles[index]
        self.particles = p_copy
        self.weights = np.ones(config.filter.n_particles, dtype=np.float32) / config.filter.n_particles

    def get_degeneration_index(self):
        return 1 / np.sum(self.weights ** 2)

    def eval_policy(self, index):
        policy = self.particles[index] if index >= 0 else None
        self.test_iter.particles = policy
        #print("policy:", policy)
        [_, _, total_loss] = validate(self.test_loader, self.test_iter, self.model, self.loss, suppress_output=True, calc_acc=False)
        if type(self.test_loader) == DALIClassificationIterator or type(self.test_loader) == DALIGenericIterator:
            self.test_loader.reset()
        return total_loss

    def p_enter(self, epoch=0, enabled=True):
        self.epoch = epoch
        if not enabled or self.epoch % config.filter.interval != 0:
            return
        start = time.time()
        logger.info("Epoch: [{}/{}] Filter step part 1/2".format(self.epoch + 1, config.epochs))
        self.add_noise()
        if config.distributed:
            broadcast_numpy_tensor(self.particles)
        for i in range(-1, config.filter.n_particles):
            self.old_losses[i] = self.eval_policy(i)
        self.train_iter.particles = self.particles
        self.train_iter.weights = self.weights
        logger.info("Epoch: [{}/{}] Filter step part 1/2 elapsed: {}".format(self.epoch + 1, config.epochs, time.time() - start))

        
    def p_exit(self, epoch=0, enabled=True):
        self.epoch = epoch
        if not enabled or self.epoch % config.filter.interval != 0:
            return
        start = time.time()
        logger.info("Epoch: [{}/{}] Filter step part 2/2".format(self.epoch + 1, config.epochs))
        for i in range(-1, config.filter.n_particles):
            self.new_losses[i] = self.eval_policy(i)
        diff = (self.old_losses - self.new_losses) / config.batch_size
        diff = diff[:-1] / diff[-1]
        diff = np.tanh(diff - 1) + 1
        self.weights *= np.power(diff, config.filter.lr)
        self.nomalize_weights()
        if self.get_degeneration_index() < config.filter.n_particles * config.filter.resample_threshold:
            self.resample()
        if config.filter.only_good:
            good_policy_ids = self.weights * self.weights.shape[0] > config.filter.good_threshold
            particles = np.array([p for i, p in enumerate(self.particles) if good_policy_ids[i]])
            weights = self.weights[good_policy_ids]
            weights = weights / np.sum(weights)
            weights[-1] = 1 - np.sum(weights[:-1])
            weights = np.clip(weights, 0.0, 1.0)
            self.train_iter.particles = particles
            self.train_iter.weights = weights
        else:
            self.train_iter.particles = self.particles
            self.train_iter.weights = self.weights
        if self.epoch % config.filter.print_policies_every == 0:
            print_current_policies(self.particles, self.weights)
        logger.info("Epoch: [{}/{}] Filter step part 2/2 elapsed: {}".format(self.epoch + 1, config.epochs, time.time() - start))


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

def broadcast_numpy_tensor(np_tensor):
    torch_tensor = torch.from_numpy(np_tensor).to(config.local_rank)
    dist.broadcast(torch_tensor, 0)
    if config.local_rank > 0:
        np_tensor = torch_tensor.cpu().numpy()
    return np_tensor

def print_current_policies(particles, weights, threshold=0.2):
    i_max = np.argmax(weights)
    best_policy = particles[i_max]
    logger.info("Best policy: [" + ", ".join(["{:.2f}".format(p) for p in best_policy]) + "] ({})".format(i_max + 1))
    logger.info("Weight degeneracy index: {:.2f}".format(1 / np.sum(weights ** 2) / weights.shape[0]))
    for i in range(weights.shape[0]):
        logger.info("Policy {} (p/pavg={:.2f}): [".format(i + 1, weights[i] * weights.shape[0]) + ", ".join(["{:.2f}".format(p) for p in particles[i]]) + "]")

def merge_dict_entry(dst, patch):
    for key in patch.keys():
        if key not in dst.keys():
            dst[key] = patch[key]
        else:
            if type(patch[key]) == dict:
                merge_dict_entry(dst[key], patch[key])
            else:
                if patch[key] == "None":
                    patch[key] = None
                dst[key] = patch[key]

# From https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
def merge_dict(a, b, path=None):
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                 a[key] = b[key]
        else:
            a[key] = b[key]
    return a

if __name__ == '__main__':
    if len(sys.argv) > 2:
        overrides = dict()
        for kv in sys.argv[2:]:
            temp_dict = dict()
            k, v = kv.split("=")
            k = k.split(".")
            try:
                v = float(v)
            except ValueError:
                pass
            if type(v) == float and v.is_integer():
                v = int(v)
            if v == "True":
                v = True
            if v == "False":
                v = False
            if len(k) == 1:
                temp_dict[k[0]] = v
            else:
                cur_dict = {k[-1]: v}
                for i in range(len(k) - 2, 0, -1):
                    cur_dict = {k[i]: cur_dict}
                temp_dict[k[0]] = cur_dict
            merge_dict_entry(overrides, temp_dict)
    else:
        overrides = None

    folder_name = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    os.chdir(folder_name)
    if len(glob.glob("*")) > 0:
        Path("backup").mkdir(parents=True, exist_ok=True)
        for file in glob.glob("*"):
            if file != "backup":
                shutil.move(file, os.path.join("backup", file))

    if len(sys.argv) > 1 and int(sys.argv[1]) > 1:
        torchmp.start_processes(main, nprocs=int(sys.argv[1]), args=(int(sys.argv[1]), overrides), start_method="forkserver")
    else:
        main(overrides=overrides)