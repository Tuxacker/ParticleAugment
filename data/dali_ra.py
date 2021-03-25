from functools import partial
import io
import os
import pickle

import lmdb
import numpy as np

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import LastBatchPolicy

class LMDBClassIter:

    def __init__(self, path, batch_size, shuffle=True, partial_batch_policy=LastBatchPolicy.DROP, last_batch_padded=False, sublist=None):
        self.path = os.path.expanduser(path)
        self.batch_size = batch_size
        self.env = lmdb.open(self.path, subdir=os.path.isdir(self.path), readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__")) if sublist is None else len(sublist)
        self.shuffle = shuffle
        self.sublist = sublist
        self.partial_batch_policy = partial_batch_policy
        self.last_batch_padded = last_batch_padded
        self.raise_stop_after = False
        

    def __iter__(self):
        # print("Called __iter__")
        self.i = 0
        self.raise_stop_after = False
        if self.shuffle:
            if self.sublist is not None:
                self.shuffle_indices = np.random.permutation(self.sublist)
            else:
                self.shuffle_indices = np.random.permutation(self.length)
        return self

    def __next__(self):

        if self.raise_stop_after or (self.i + self.batch_size - 1 >= self.length and self.partial_batch_policy == LastBatchPolicy.DROP):
            if not self.last_batch_padded: 
                start_index = self.length % self.batch_size # TODO: Check last_batch_padded_behavior
                self.__iter__()
                self.i = start_index
            else:
                self.__iter__()
            # print("Raising StopIteration")
            raise StopIteration

        batch = []
        labels = []

        # print("Started iteration with start index: ", self.i)

        for _ in range(self.batch_size):

            if self.i < self.length:
                cur_index = self.i
            elif self.partial_batch_policy == LastBatchPolicy.PARTIAL:
                self.raise_stop_after = True
                break
            elif self.last_batch_padded:
                self.raise_stop_after = True
                cur_index = self.length - 1
            else:
                self.raise_stop_after = True
                cur_index = self.i % self.length

            if self.shuffle:
                    cur_index = self.shuffle_indices[cur_index]
            elif self.sublist is not None:
                cur_index = self.sublist[cur_index]

            with self.env.begin(write=False) as txn:
                # print("Index: ", cur_index)
                byteflow = txn.get("{}".format(cur_index).encode("ascii"))
            image, label = pickle.loads(byteflow)

            with io.BytesIO() as arr:
                arr.write(image)
                arr.seek(0)
                image = np.load(arr, allow_pickle=True)
            batch.append(image)
            labels.append(np.array(label))
            self.i += 1

        return (batch, labels)

    def __len__(self):
        return self.length

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

        def get_enhanced_fparam(probability):
            signs = fn.random.coin_flip(probability=0.5) * 2 - 1
            to_apply = fn.random.coin_flip(probability=probability)
            return (self.get_fparam(0.9) * signs + 1) * to_apply + 1.0 * (to_apply ^ True)# Lower bound maybe?

        self.get_enhanced_fparam = get_enhanced_fparam

        def blend(img1, img2, alpha):
            return (1 - alpha) * img1 + alpha * img2

        self.blend = blend

        def mux(cond, aug_img, orig_img):
            neg_cond = cond ^ True
            return cond * aug_img + neg_cond * orig_img

        self.mux = mux

        # ShearX
        self.shearxp = partial(fn.warp_affine, device="gpu", matrix=(1, self.get_fparam(0.3), 0, 0, 1, 0), inverse_map=False)
        self.shearxn = partial(fn.warp_affine, device="gpu", matrix=(1, -self.get_fparam(0.3), 0, 0, 1, 0), inverse_map=False)

        # ShearY
        self.shearyp = partial(fn.warp_affine, device="gpu", matrix=(1, 0, 0, self.get_fparam(0.3), 1, 0), inverse_map=False)
        self.shearyn = partial(fn.warp_affine, device="gpu", matrix=(1, 0, 0, -self.get_fparam(0.3), 1, 0), inverse_map=False)

        # TranslateX
        self.trxp = partial(fn.warp_affine, device="gpu", matrix=(1, 0, self.get_iparam(10), 0, 1, 0), inverse_map=False)
        self.trxn = partial(fn.warp_affine, device="gpu", matrix=(1, 0, -self.get_iparam(10), 0, 1, 0), inverse_map=False)

        # TranslateY
        self.tryp = partial(fn.warp_affine, device="gpu", matrix=(1, 0, 0, 0, 1, self.get_iparam(10)), inverse_map=False)
        self.tryn = partial(fn.warp_affine, device="gpu", matrix=(1, 0, 0, 0, 1, -self.get_iparam(10)), inverse_map=False)

        # RandomCrop
        self.crop = partial(fn.random_resized_crop, device="gpu", size=[32, 32], random_area=[float((7/8)**2), 1.0])

        # Thumbnail
        self.thumbnail = partial(fn.resize, device="gpu", size=[8, 8])

        # UINT8->F32
        self.to_float = partial(fn.cast, device="gpu", dtype=types.DALIDataType.FLOAT)
        self.to_bchw = partial(fn.transpose, device="gpu", perm=[2, 0, 1]) # axes are already in "WH" format
        self.normalize = partial(fn.normalize, device="gpu", axes=[0, 1], batch=True) #, mean=[0.4914, 0.4822, 0.4465], stddev=[0.2470, 0.2435, 0.2616])


        




    def define_ra_graph(self, input, labels):

        images = input.gpu()
        labels = labels.gpu()

        # Rotation
        angle = self.get_fparam(30.0) * fn.random.coin_flip(probability=0.1)
        images_0 = fn.rotate(images, angle=angle, device="gpu", keep_size=True)

        # Saturation
        sat = self.get_enhanced_fparam(0.1)
        images_1 = fn.hsv(images_0, saturation=sat, device="gpu")

        # Contrast
        cont = self.get_enhanced_fparam(0.1)
        images_2 = fn.contrast(images_1, contrast=cont, device="gpu")

        # Brightness
        brt = self.get_enhanced_fparam(0.1)
        images_3 = fn.brightness(images_2, brightness=brt, device="gpu")

        # Sharpness (Converted to Blur temporarily)
        shp = self.get_fparam(-0.9) * fn.random.coin_flip(probability=0.1) + 1 #self.get_enhanced_fparam(1.0)
        images_4 = self.blend(fn.gaussian_blur(images_3, window_size=3, device="gpu"), images_3, shp)

        # ShearX
        sx = fn.random.coin_flip(probability=0.1)
        sxsign = fn.random.coin_flip(probability=0.5)
        img_sx = self.mux(sxsign, self.shearxp(images_4), self.shearxn(images_4))
        images_5 = self.mux(sx, img_sx, images_4)

        # ShearY
        sy = fn.random.coin_flip(probability=0.1)
        sysign = fn.random.coin_flip(probability=0.5)
        img_sy = self.mux(sysign, self.shearyp(images_5), self.shearyn(images_5))
        images_6 = self.mux(sy, img_sy, images_5)

        # TranslateX
        tx = fn.random.coin_flip(probability=0.1)
        txsign = fn.random.coin_flip(probability=0.5)
        img_tx = self.mux(txsign, self.trxp(images_6), self.trxn(images_6))
        images_7 = self.mux(tx, img_tx, images_6)

        # TranslateY
        ty = fn.random.coin_flip(probability=0.1)
        tysign = fn.random.coin_flip(probability=0.5)
        img_ty = self.mux(tysign, self.tryp(images_7), self.tryn(images_7))
        images_8 = self.mux(ty, img_ty, images_7)

        # Random Crop
        images_9 = self.crop(images_8)

        to_flip = fn.random.coin_flip(probability=0.5)
        images_10 = fn.flip(images_9, horizontal=to_flip, device="gpu")

        images_thumb = self.thumbnail(images)
        thumb_patches = fn.paste(images_thumb, paste_x=fn.random.uniform(range=[0.0, 0.75]), paste_y=fn.random.uniform(range=[0.0, 0.75]), ratio=4, fill_value=0, device="gpu") 
        thumb_mask = thumb_patches > 0
        thumb_blend = self.mux(thumb_mask, thumb_patches, images_10)
        use_thumb = fn.random.coin_flip(probability=0.1)
        images_11 = self.mux(use_thumb, thumb_blend, images_10)

        # Convert to typical PyTorch input
        images_float = self.to_float(images_11) / 255
        images_norm = self.normalize(images_float)
        images_final = self.to_bchw(images_norm)

        return images_final, labels


class LCRAPipeline(RAPipeline):

    def __init__(self, path, batch_size, num_threads, device_id, num_gpus, magnitude, shuffle=True, last_batch_policy=LastBatchPolicy.DROP, last_batch_padded=False, sublist=None, max_magnitude=30):
        super(LCRAPipeline, self).__init__(batch_size, num_threads, device_id, num_gpus, magnitude, max_magnitude)

        self.iterator = LMDBClassIter(path, batch_size, shuffle=shuffle, partial_batch_policy=last_batch_policy, last_batch_padded=last_batch_padded, sublist=sublist)

    def define_graph(self):
        images, labels = fn.external_source(self.iterator, num_outputs=2, cycle="raise")
        images_out, labels_out = self.define_ra_graph(images, labels)
        return images_out, labels_out
