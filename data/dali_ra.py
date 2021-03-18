import io
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
                    byteflow = txn.get("{}".format(self.shuffle_indices[self.i]).encode("ascii"))
                else:
                    byteflow = txn.get("{}".format(self.i).encode("ascii"))
            image, label = pickle.loads(byteflow)

            with io.BytesIO() as arr:
                arr.write(image)
                arr.seek(0)
                image = np.load(arr, allow_pickle=True)
            batch.append(image)
            labels.append(np.array(label))
            self.i += 1
            if self.i == self.length:
                raise StopIteration # TODO: Implement proper iterator behavior
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
            return float(param * self.magnitude / self.max_magnitude)

        self.get_fparam = get_fparam

        def get_iparam(param=10):
            return int(param * self.magnitude / self.max_magnitude)

        self.get_iparam = get_iparam

        self.rng = ops.random.CoinFlip()

        #self.shape = ops.Shapes(dtype=types.DALIDataType.INT32)

        def get_enhanced_fparam(probability):
            signs = self.rng(probability=0.5) * 2 - 1
            to_apply = self.rng(probability=probability)
            return (self.get_fparam(0.9) * signs + 1) * to_apply + 1.0 * (to_apply ^ True)# Lower bound maybe?

        self.get_enhanced_fparam = get_enhanced_fparam

        def blend(img1, img2, alpha):
            return (1 - alpha) * img1 + alpha * img2

        self.blend = blend

        def mux(cond, aug_img, orig_img):
            neg_cond = cond ^ True
            return cond * aug_img + neg_cond * orig_img

        self.mux = mux

        # Rotation
        self.rotate = ops.Rotate(device="gpu", keep_size=True)

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
        self.shearxn = ops.WarpAffine(device="gpu", matrix=(1, -self.get_fparam(0.3), 0, 0, 1, 0), inverse_map=False)

        # ShearY
        self.shearyp = ops.WarpAffine(device="gpu", matrix=(1, 0, 0, self.get_fparam(0.3), 1, 0), inverse_map=False)
        self.shearyn = ops.WarpAffine(device="gpu", matrix=(1, 0, 0, -self.get_fparam(0.3), 1, 0), inverse_map=False)

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
        images_0 = self.rotate(images, angle=angle)

        # Saturation
        sat = self.get_enhanced_fparam(0.1)
        images_1 = self.saturation(images_0, saturation=sat)

        # Contrast
        cont = self.get_enhanced_fparam(0.1)
        images_2 = self.contrast(images_1, contrast=cont)

        # Brightness
        brt = self.get_enhanced_fparam(0.1)
        images_3 = self.brightness(images_2, brightness=brt)

        # Sharpness (Converted to Blur temporarily)
        shp = self.get_fparam(-0.9) * self.rng(probability=0.1) + 1 #self.get_enhanced_fparam(1.0)
        images_4 = self.blend(self.sharpness(images_3), images_3, shp)

        # ShearX
        sx = self.rng(probability=0.1)
        sxsign = self.rng(probability=0.5)
        img_sx = self.mux(sxsign, self.shearxp(images_4), self.shearxn(images_4))
        images_5 = self.mux(sx, img_sx, images_4)

        # ShearY
        sy = self.rng(probability=0.1)
        sysign = self.rng(probability=0.5)
        img_sy = self.mux(sysign, self.shearyp(images_5), self.shearyn(images_5))
        images_6 = self.mux(sy, img_sy, images_5)

        # TranslateX
        tx = self.rng(probability=0.1)
        txsign = self.rng(probability=0.5)
        img_tx = self.mux(txsign, self.trxp(images_6), self.trxn(images_6))
        images_7 = self.mux(tx, img_tx, images_6)

        # TransalteY
        ty = self.rng(probability=0.1)
        tysign = self.rng(probability=0.5)
        img_ty = self.mux(tysign, self.tryp(images_7), self.tryn(images_7))
        images_8 = self.mux(ty, img_ty, images_7)

        return images_8, labels


class LCRAPipeline(RAPipeline):

    def __init__(self, path, batch_size, num_threads, device_id, num_gpus, magnitude, max_magnitude=30):
        super(LCRAPipeline, self).__init__(batch_size, num_threads, device_id, num_gpus, magnitude, max_magnitude)

        self.iterator = LMDBClassIter(path, batch_size)
        self.input = ops.ExternalSource(self.iterator, num_outputs=2, cycle="raise")

    def define_graph(self):
        images, labels = self.input()
        images_out, labels_out = self.define_ra_graph(images, labels)
        return images_out, labels_out
