import os
import pickle

import lmdb
import numpy as np

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.fn as fn

class LMDBClassIter:

    def __init__(self, path, batch_size, shuffle=True, shuffle_each_epoch=True, partial_batch=False):
        self.path = os.path.expanduser(path)
        self.batch_size = batch_size
        self.env = lmdb.open(self.path, subdir=os.path.isdir(self.path), readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
        self.shuffle = shuffle
        self.shuffle_each_epoch = shuffle_each_epoch
        self.partial_batch = partial_batch
        if self.shuffle:
            self.shuffle_indices = np.random.permutation(self.length)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            with self.env.begin(write=False) as txn:
                if self.shuffle:
                    byteflow = txn.get(b"{}".format(self.shuffle_indices[self.i]).encode("ascii"))
                else:
                    byteflow = txn.get(b"{}".format(self.i).encode("ascii"))
            image, label = pickle.loads(byteflow)
            batch.append(image)
            labels.append(label)
            self.i += 1
            if self.i == self.length:
                if self.shuffle and self.shuffle_each_epoch:
                    self.shuffle_indices = np.random.permutation(self.length)
                self.i = 0
                # TODO: Check how shuffling is handled in last partial batch and how partial batches are handled in NVIDIA DALI
                if self.partial_batch:
                    break
        return (batch, labels)

def mux(cond, aug_img, orig_img):
    neg_cond = cond ^ True
    return cond * aug_img + neg_cond * orig_img

class RAPipeline(Pipeline):

    def __init__(self, batch_size, num_threads, device_id, num_gpus, magnitude, max_magnitude=30):
        super(RAPipeline, self).__init__(batch_size, num_threads, device_id)

        self.magnitude = magnitude
        self.max_magnitude = max_magnitude

        def get_fparam(param=1.0):
            return float(param * magnitude / self.max_magnitude)

        def get_enhanced_fparam():
            return float(1.8 * magnitude / self.max_magnitude) + 0.1

        self.get_fparam = get_fparam
        self.get_enhanced_fparam = get_enhanced_fparam

        self.angle = ops.random.CoinFlip()
        self.rotate = ops.Rotate(device="gpu")

    def define_ra_graph(self, input, labels):
        angle = self.get_fparam(30.0) * self.angle(shape=(self.max_batch_size))
        #angle = fn.cast(angle, dtype=types.FLOAT)
        images = self.rotate(input.gpu(), angle=angle) # TODO: Check if input is on CPU or GPU
        return images, labels


class LCRAPipeline(RAPipeline):

    def __init__(self, path, batch_size, num_threads, device_id, num_gpus, magnitude, max_magnitude=30):
        super(LCRAPipeline, self).__init__(batch_size, num_threads, device_id, num_gpus, magnitude, max_magnitude)

        self.iterator = LMDBClassIter(path, batch_size)
        self.input = ops.ExternalSource(self.iterator, num_outputs=2)

    def define_graph(self):
        images, labels = self.input()
        images, labels = self.define_ra_graph(images, labels)
        return images, labels
