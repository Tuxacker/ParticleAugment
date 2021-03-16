import os
import pickle

import lmdb
import numpy as np

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
#import nvidia.dali.fn as fn

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

class RAPipeline(Pipeline):

    def __init__(self, batch_size, num_threads, device_id, num_gpus, magnitude, max_magnitude=30):
        super(RAPipeline, self).__init__(batch_size, num_threads, device_id)

        self.magnitude = magnitude
        self.max_magnitude = max_magnitude

        def get_fparam(param=1.0):
            return float(param * magnitude / self.max_magnitude)

        self.get_fparam = get_fparam

        def get_iparam(param=10):
            return int(param * magnitude / self.max_magnitude)

        self.get_iparam = get_iparam

        self.rng = ops.random.CoinFlip()

        def get_enhanced_fparam(input, probability):
            signs = self.rng(input, probability=0.5) * 2 - 1
            to_apply = self.rng(input, probability=probability)
            return (self.get_fparam(0.9) * signs + 1) * to_apply # Lower bound maybe?

        self.get_enhanced_fparam = get_enhanced_fparam

        def blend(img1, img2, alpha):
            return (1 - alpha) * img1 + alpha * img2

        self.blend = blend

        def mux(cond, aug_img, orig_img):
            neg_cond = cond ^ True
            return cond * aug_img + neg_cond * orig_img

        self.mux = mux

        # Rotation
        self.rotate = ops.Rotate(device="gpu")

        # Saturation
        self.saturation = ops.Hsv(device="gpu")

        # Contrast
        self.contrast = ops.Contrast(device="gpu")

        # Brightness
        self.brightness = ops.Brightness(device="gpu")

        # Sharpness
        self.sharpness = ops.GaussianBlur(device = "gpu", window_size=3)

        # ShearX
        self.shearxp = ops.WarpAffine(device="gpu", matrix=(1, self.get_fparam(0.3), 0, 0, 1, 0), inverse_map=False)
        self.shearxn = ops.WarpAffine(device="gpu", matrix=(1, self.get_fparam(0.3), 0, 0, 1, 0), inverse_map=False)

        # ShearY
        self.shearyp = ops.WarpAffine(device="gpu", matrix=(1, 0, 0, self.get_fparam(0.3), 1, 0), inverse_map=False)
        self.shearyn = ops.WarpAffine(device="gpu", matrix=(1, 0, 0, self.get_fparam(0.3), 1, 0), inverse_map=False)

        # TranslateX
        self.trxp = ops.WarpAffine(device="gpu", matrix=(1, 0, self.get_iparam(10), 0, 1, 0), inverse_map=False)
        self.trxn = ops.WarpAffine(device="gpu", matrix=(1, 0, -self.get_iparam(10), 0, 1, 0), inverse_map=False)

        # TranslateY
        self.tryp = ops.WarpAffine(device="gpu", matrix=(1, 0, 0, 0, 1, self.get_iparam(10)), inverse_map=False)
        self.tryn = ops.WarpAffine(device="gpu", matrix=(1, 0, 0, 0, 1, -self.get_iparam(10)), inverse_map=False)




    def define_ra_graph(self, input, labels):

        images = input.gpu()
        labels = labels.gpu()

        # Rotation
        angle = self.get_fparam(30.0) * self.rng(probability=0.1)
        images = self.rotate(images, angle=angle) # TODO: Check if input is on CPU or GPU

        # Saturation
        sat = self.get_enhanced_fparam(input, 0.1)
        images = self.saturation(images, saturation=sat)

        # Contrast
        cont = self.get_enhanced_fparam(input, 0.1)
        images = self.contrast(images, contrast=cont)

        # Brightness
        brt = self.get_enhanced_fparam(input, 0.1)
        images = self.brightness(images, brightness=brt)

        # Sharpness
        shp = self.get_enhanced_fparam(input, 0.1)
        images = self.blend(self.sharpness(images), images, shp)

        # ShearX
        sx = self.rng(input, probability=0.1)
        sxsign = self.rng(input, probability=0.5) * 2 - 1
        img_sx = self.mux(sxsign, self.shearxp(images), self.shearxn(images))
        images = self.mux(sx, img_sx, images)

        # ShearY
        sy = self.rng(input, probability=0.1)
        sysign = self.rng(input, probability=0.5) * 2 - 1
        img_sy = self.mux(sysign, self.shearyp(images), self.shearyn(images))
        images = self.mux(sy, img_sy, images)

        # TranslateX
        tx = self.rng(input, probability=0.1)
        txsign = self.rng(input, probability=0.5) * 2 - 1
        img_tx = self.mux(txsign, self.trxp(images), self.trxn(images))
        images = self.mux(tx, img_tx, images)

        # TransalteY
        ty = self.rng(input, probability=0.1)
        tysign = self.rng(input, probability=0.5) * 2 - 1
        img_ty = self.mux(tysign, self.tryp(images), self.tryn(images))
        images = self.mux(ty, img_ty, images)

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
