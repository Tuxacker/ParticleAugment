from functools import partial
import io
import os
import pickle

import lmdb
import numpy as np
#import cupy
#from cupyx.scipy.signal import convolve2d

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import nvidia.dali.math as dalimath
from nvidia.dali.plugin.pytorch import LastBatchPolicy

from numba import njit

@njit
def autocontrast(img):
    for i in range(3):
        black = np.min(img[:, :, i])
        white = np.max(img[:, :, i])
        img[:, :, i] = ((img[:, :, i] - black) * 255.0 /(white - black)).astype(np.uint8)
    return img

@njit
def equalize(img):
    size = img.shape[0] * img.shape[1]
    for i in range(3):
        h, _ = np.histogram(img[:, :, i].flatten(), bins=256, range=(0, 255))
        csum = np.cumsum(h)
        cmin = csum[(csum > 0).argmax()]
        divisor = size - cmin
        if divisor == 0:
            continue
        for k in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[k, j, i] = round((csum[img[k, j, i]] - cmin) / divisor * 255.0)
    return img

@njit
def random_erase(img):
    area = img.shape[0] * img.shape[1]
    for _ in range(10):
        target_area = np.random.uniform(0.02, 1/3) * area
        aspect_ratio = np.exp(np.random.uniform(-1.204, 1.204))
        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))
        if w < img.shape[1] and h < img.shape[0]:
            top = np.random.randint(0, img.shape[0] - h)
            left = np.random.randint(0, img.shape[1] - w)
            img[top:top + h, left:left + w] = 0
            break
    return img

@njit
def random_crop(img, padding=4):
    paste = np.zeros((img.shape[0] + padding, img.shape[1] + padding, 3), dtype=np.uint8)
    paste[padding:padding+img.shape[0], padding:padding+img.shape[1]] = img
    t = np.random.randint(0, paste.shape[0] - img.shape[0] + 1)
    l = np.random.randint(0, paste.shape[1] - img.shape[1] + 1)
    img = paste[t:t+img.shape[0],l:l+img.shape[1]]
    return img


#@njit
#def posterize(img, bits):
#    return img & ~(2 ** (8-bits) - 1)

#@njit
#def invert(img):
#    return 255 - img

#@njit
#def solarize(img):
#    for i in range(3):
#        for j in range(img.shape[1]):
#            mask = np.greater_equal(img[:, j, i], 128)
#            img[mask, j, i] = 255 - img[mask, j, i] 
#    return img

#@njit
#def solarize_add(img, add):
#    for i in range(3):
#        for j in range(img.shape[1]):
#            mask = np.less(img[:, j, i], 128)
#            img[mask, j, i] = np.min(255, img[mask, j, i] + add) 
#    return img



class LMDBClassIter:

    def __init__(self, path, batch_size, device_id=0, num_gpus=1, shuffle=True, last_batch_policy=LastBatchPolicy.DROP, last_batch_padded=False, sublist=None, particles=None, weights=None, output_augmentations=True, seed=0):
        self.path = os.path.expanduser(path)
        self.batch_size = batch_size
        self.env = lmdb.open(self.path, subdir=os.path.isdir(self.path), readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__")) if sublist is None else len(sublist)
        self.original_length = self.length
        self.shuffle = shuffle
        self.sublist = sublist
        self.original_sublist = sublist
        self.device_id = device_id
        self.num_gpus = num_gpus
        if num_gpus > 1:
            if self.length % num_gpus != 0:
                print("Warning: Dataset length not divisible by the number of GPUs, truncating...")
                self.length = self.length - self.length % num_gpus
                self.original_length = self.length
                if self.original_sublist is not None:
                    self.original_sublist = self.original_sublist[:self.length]
            self.shard_size = self.length // self.num_gpus
        self.last_batch_policy = last_batch_policy
        self.last_batch_padded = last_batch_padded
        self.raise_stop_after = False
        self.epoch = 0
        self.particles = particles
        self.weights = weights
        self.ra_base = np.zeros(12, dtype=np.float32)
        self.ra_base[0:2] = 1.0
        self.output_augmentations = output_augmentations
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        #self.name = "default"
        

    def __iter__(self):
        # print("Called __iter__")
        self.i = 0
        if self.num_gpus > 1:
            if self.shuffle:
                self.rng = np.random.default_rng(self.seed + self.epoch)
                #start_index = ((self.device_id + self.epoch) % self.num_gpus) * self.original_length // self.num_gpus
                #end_index = ((self.device_id + self.epoch) % self.num_gpus + 1) * self.original_length // self.num_gpus
                if self.original_sublist is not None:
                    self.sublist = self.rng.permutation(self.original_sublist)[self.device_id:self.original_length:self.num_gpus]
                    #self.sublist = self.original_sublist[start_index:end_index]
                    #print("case 1:", start_index, ":", end_index, "@", self.epoch, ",", self.device_id)
                else:
                    self.sublist = self.rng.permutation(self.original_length)[self.device_id:self.original_length:self.num_gpus]
                    #self.sublist = np.arange(start_index, end_index, dtype=np.int32)
                    #print("case 2:", start_index, ":", end_index, "@", self.epoch, ",", self.device_id)
                self.length = len(self.sublist)
            else:
                if self.original_sublist is not None:
                    self.sublist = self.original_sublist[self.device_id:self.original_length:self.num_gpus]
                    #self.sublist = self.original_sublist[start_index:end_index]
                    #print("case 1:", start_index, ":", end_index, "@", self.epoch, ",", self.device_id)
                else:
                    self.sublist = np.arange(self.device_id, self.original_length, self.num_gpus)
                    #self.sublist = np.arange(start_index, end_index, dtype=np.int32)
                    #print("case 2:", start_index, ":", end_index, "@", self.epoch, ",", self.device_id)
                self.length = len(self.sublist)
            #print("Sublist:{} {}-{}".format(self.sublist, start_index, end_index))
        self.raise_stop_after = False
        if self.shuffle:
            if self.original_sublist is not None:
                self.sublist = self.rng.permutation(self.original_sublist)
            else:
                self.sublist = self.rng.permutation(self.length)
        return self

    def __next__(self):

        #print("using: ", self.particles, "at ", self.name)

        if self.raise_stop_after or (self.i + self.batch_size - 1 >= self.length and self.last_batch_policy == LastBatchPolicy.DROP) or self.i >= self.length:
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
        augmentations = []

        # print("Started iteration with start index: ", self.i)

        for _ in range(self.batch_size):

            if self.i < self.length:
                cur_index = self.i
            elif self.last_batch_policy == LastBatchPolicy.PARTIAL:
                self.raise_stop_after = True
                break
            elif self.last_batch_padded:
                self.raise_stop_after = True
                cur_index = self.length - 1
            else:
                self.raise_stop_after = True
                cur_index = self.i % self.length

            #if self.shuffle:
            #    cur_index = self.shuffle_indices[cur_index]
            if self.sublist is not None:
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

            #print("Device {} index {}".format(self.device_id, cur_index))

            if not self.output_augmentations:
                self.i += 1
                continue

            if self.particles is not None:
                if self.particles.ndim == 1:
                    augmentations.append(self.particles)
                else:
                    index = self.rng.choice(len(self.weights), 1, p=self.weights)[0]
                    augmentations.append(self.particles[index])
            else:
                augmentations.append(self.rng.permutation(self.ra_base))

            augmentations[-1] = 1 - (augmentations[-1] / (np.sum(augmentations[-1]) + 1))

            #batch[-1] = random_crop(batch[-1]).astype(np.uint8)
            #if self.rng.random() <= 0.5:
            #    batch[-1] = random_erase(batch[-1]).astype(np.uint8)

            #if len(augmentations[-1]) == 15:
                #if self.rng.random() < augmentations[-1][-2]:
                #    batch[-1] = autocontrast(batch[-1]).astype(np.uint8)
                #if self.rng.random() < augmentations[-1][-1]:
                #    batch[-1] = equalize(batch[-1]).astype(np.uint8)
                #if self.rng.random() > augmentations[-1][-3]:
                #    batch[-1] = posterize(batch[-1], 4).astype(np.uint8)
                #if self.rng.random() > augmentations[-1][-3]:
                #    batch[-1] = invert(batch[-1])
                #if self.rng.random() > augmentations[-1][-2]:
                #    batch[-1] = solarize(batch[-1])
                #if self.rng.random() > augmentations[-1][-1]:
                #    batch[-1] = solarize_add(batch[-1], 55)

            self.i += 1

        if not self.output_augmentations:
            return (batch, labels)

        return (batch, labels, augmentations)

    def __len__(self):
        return len(self.sublist) if self.sublist is not None else self.length

    def set_epoch(self, epoch):
        self.epoch = epoch
        #print("set epoch to", self.epoch)

class RAOperations:

    def __init__(self, device, magnitude, max_magnitude=10):

        self.magnitude = magnitude
        self.max_magnitude = max_magnitude

        def get_fparam(param=1.0):
            return float(param * self.magnitude / self.max_magnitude)

        self.get_fparam = get_fparam

        def get_iparam(param=10):
            return int(param * self.magnitude / self.max_magnitude)

        self.get_iparam = get_iparam

        def get_enhanced_fparam_d():
            signs = fn.random.coin_flip(probability=0.5) * 2 - 1
            return (self.get_fparam(0.9) * signs + 1)

        self.get_enhanced_fparam_d = get_enhanced_fparam_d

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

        # Sharpness

        #cupy.cuda.Device(device).use()
        #self.smooth_kernel = 1 / 13 * cupy.array([1, 1, 1, 1, 5, 1, 1, 1, 1]).reshape((3, 3))

        #def sharpness_func(x):
        #    a1 = cupy.fromDlpack(x)
        #    result = cupy.empty_like(a1)
        #    for i in range(3): 
        #        result[i] = convolve2d(a1[i], self.smooth_kernel, mode="same")
        #    cupy.cuda.get_current_stream().synchronize()
        #    return result.toDlpack()

        #self.sharpness = sharpness_func

        # ShearX
        self.shearxp = partial(fn.warp_affine, device="gpu", matrix=(1, self.get_fparam(0.3), 0, 0, 1, 0), inverse_map=False, fill_value=128)
        self.shearxn = partial(fn.warp_affine, device="gpu", matrix=(1, -self.get_fparam(0.3), 0, 0, 1, 0), inverse_map=False, fill_value=128)

        # ShearY
        self.shearyp = partial(fn.warp_affine, device="gpu", matrix=(1, 0, 0, self.get_fparam(0.3), 1, 0), inverse_map=False, fill_value=128)
        self.shearyn = partial(fn.warp_affine, device="gpu", matrix=(1, 0, 0, -self.get_fparam(0.3), 1, 0), inverse_map=False, fill_value=128)

        # TranslateX
        self.trxp = partial(fn.warp_affine, device="gpu", matrix=(1, 0, self.get_iparam(10), 0, 1, 0), inverse_map=False, fill_value=128)
        self.trxn = partial(fn.warp_affine, device="gpu", matrix=(1, 0, -self.get_iparam(10), 0, 1, 0), inverse_map=False, fill_value=128)

        # TranslateY
        self.tryp = partial(fn.warp_affine, device="gpu", matrix=(1, 0, 0, 0, 1, self.get_iparam(10)), inverse_map=False, fill_value=128)
        self.tryn = partial(fn.warp_affine, device="gpu", matrix=(1, 0, 0, 0, 1, -self.get_iparam(10)), inverse_map=False, fill_value=128)

        # RandomCrop
        self.crop = partial(fn.random_resized_crop, device="gpu", size=[32, 32], random_area=[float((7/8)**2), 1.0])

        # Thumbnail
        self.thumbnail = partial(fn.resize, device="gpu", size=[8, 8])

        # Posterize
        self.bitmask = types.Constant(~(2 ** (8-self.get_iparam(4)) - 1)).uint8()

        # UINT8->F32
        self.to_float = partial(fn.cast, device="gpu", dtype=types.DALIDataType.FLOAT)
        self.to_bchw = partial(fn.transpose, device="gpu", perm=[2, 0, 1]) # axes are already in "WH" format
        self.normalize = partial(fn.normalize, device="gpu", axes=[0, 1], batch=True) #, mean=[0.4914, 0.4822, 0.4465], stddev=[0.2470, 0.2435, 0.2616])


def apply_cifar_pre(input, labels):
    images = input.gpu()
    labels = labels.gpu()

    # Horizontal Flip
    # images_flip = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5), device="gpu")
    images_flip = images

    return images_flip, labels

def apply_ra_single(input, labels, augmentation_list, params):

    images = input.gpu()
    labels = labels.gpu()

    augmented_images = []

    augmented_images.append(fn.rotate(images, angle=params.get_fparam(30.0), device="gpu", keep_size=True, fill_value=128))

    augmented_images.append(fn.hsv(images, saturation=params.get_enhanced_fparam_d(), device="gpu"))

    augmented_images.append(fn.contrast(images, contrast=params.get_enhanced_fparam_d(), device="gpu"))

    augmented_images.append(fn.brightness(images, brightness=params.get_enhanced_fparam_d(), device="gpu"))

    sxsign = fn.random.coin_flip(probability=0.5)
    augmented_images.append(params.mux(sxsign, params.shearxp(images), params.shearxn(images)))

    sysign = fn.random.coin_flip(probability=0.5)
    augmented_images.append(params.mux(sysign, params.shearyp(images), params.shearyn(images)))

    txsign = fn.random.coin_flip(probability=0.5)
    augmented_images.append(params.mux(txsign, params.trxp(images), params.trxn(images)))

    tysign = fn.random.coin_flip(probability=0.5)
    augmented_images.append(params.mux(tysign, params.tryp(images), params.tryn(images)))

    augmented_images.append(255 - images)

    sol_mask = images > 127
    augmented_images.append(params.mux(sol_mask, augmented_images[-1], images))

    augmented_images.append(images & params.bitmask)

    images_float = params.to_float(images) / 255

    images_shp = params.blend(fn.gaussian_blur(images_float, window_size=3, device="gpu"), images_float, params.get_enhanced_fparam_d())
    augmented_images.append(fn.cast(dalimath.clamp(images_shp, 0.0, 1.0) * 255.0, device="gpu", dtype=types.DALIDataType.UINT8))

    for i in range(12):
        weights = fn.element_extract(augmentation_list, element_map=i)
        images = params.blend(augmented_images[i], images, weights)

    images_float = params.to_float(images) / 255

    images_final = fn.crop_mirror_normalize(images_float, device="gpu", mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616], output_layout="CHW")

    return images_final, labels


def apply_ra(input, labels, augmentation_list, params):

    images, labels = apply_cifar_pre(input, labels)

    augmentations = []
    for i in range(14):
        augmentations.append(fn.element_extract(augmentation_list, element_map=i))

    #augmentations_1 = fn.element_extract(augmentations, element_map=0)
    #augmentations_2 = fn.element_extract(augmentations, element_map=1)
    #augmentations_3 = fn.element_extract(augmentations, element_map=2)
    #augmentations_4 = fn.element_extract(augmentations, element_map=3)
    #augmentations_5 = fn.element_extract(augmentations, element_map=4)
    #augmentations_6 = fn.element_extract(augmentations, element_map=5)
    #augmentations_7 = fn.element_extract(augmentations, element_map=6)
    #augmentations_8 = fn.element_extract(augmentations, element_map=7)
    #augmentations_9 = fn.element_extract(augmentations, element_map=8)
    #augmentations_10 = fn.element_extract(augmentations, element_map=9)
    #augmentations_11 = fn.element_extract(augmentations, element_map=10)
    #augmentations_12 = fn.element_extract(augmentations, element_map=11)
    #augmentations_13 = fn.element_extract(augmentations, element_map=12)
    #augmentations_14 = fn.element_extract(augmentations, element_map=13)

    # Rotation
    angle = params.get_fparam(30.0) * fn.random.coin_flip(probability=augmentations[0])
    images_0 = fn.rotate(images, angle=angle, device="gpu", keep_size=True, fill_value=128)

    # Saturation
    sat = params.get_enhanced_fparam(augmentations[1])
    images_1 = fn.hsv(images_0, saturation=sat, device="gpu")

    # Contrast
    cont = params.get_enhanced_fparam(augmentations[2])
    images_2 = fn.contrast(images_1, contrast=cont, device="gpu")

    # Brightness
    brt = params.get_enhanced_fparam(augmentations[3])
    images_4 = fn.brightness(images_2, brightness=brt, device="gpu")

    # ShearX
    sx = fn.random.coin_flip(probability=augmentations[5])
    sxsign = fn.random.coin_flip(probability=0.5)
    img_sx = params.mux(sxsign, params.shearxp(images_4), params.shearxn(images_4))
    images_5 = params.mux(sx, img_sx, images_4)

    # ShearY
    sy = fn.random.coin_flip(probability=augmentations[6])
    sysign = fn.random.coin_flip(probability=0.5)
    img_sy = params.mux(sysign, params.shearyp(images_5), params.shearyn(images_5))
    images_6 = params.mux(sy, img_sy, images_5)

    # TranslateX
    tx = fn.random.coin_flip(probability=augmentations[7])
    txsign = fn.random.coin_flip(probability=0.5)
    img_tx = params.mux(txsign, params.trxp(images_6), params.trxn(images_6))
    images_7 = params.mux(tx, img_tx, images_6)

    # TranslateY
    ty = fn.random.coin_flip(probability=augmentations[8])
    tysign = fn.random.coin_flip(probability=0.5)
    img_ty = params.mux(tysign, params.tryp(images_7), params.tryn(images_7))
    images_8 = fn.cast(params.mux(ty, img_ty, images_7), dtype=types.DALIDataType.UINT8)

    # Random Crop
    images_11 = images_8 # params.crop(images_8)

    #images_thumb = params.thumbnail(images)
    #thumb_patches = fn.paste(images_thumb, paste_x=fn.random.uniform(range=[0.0, 0.75]), paste_y=fn.random.uniform(range=[0.0, 0.75]), ratio=4, fill_value=0, device="gpu") 
    #thumb_mask = thumb_patches > 0
    #thumb_blend = params.mux(thumb_mask, thumb_patches, images_9)
    #use_thumb = fn.random.coin_flip(probability=augmentations_10)
    #images_11 = params.mux(use_thumb, thumb_blend, images_9)

    # Invert
    images_inverted = 255 - images_11
    use_inv = fn.random.coin_flip(probability=augmentations[10])
    images_12 = params.mux(use_inv, images_inverted, images_11)

    # Solarize
    images_inverted_2 = 255 - images_12
    sol_mask = images_12 > 127
    images_sol = params.mux(sol_mask, images_inverted_2, images_12)

    use_sol = fn.random.coin_flip(probability=augmentations[11])
    images_13 = params.mux(use_sol, images_sol, images_12)

    # Posterize
    images_posterized =  fn.cast(images_13, device="gpu", dtype=types.DALIDataType.UINT8) & params.bitmask
    use_posterize = fn.random.coin_flip(probability=augmentations[12])
    images_14 = params.mux(use_posterize, images_posterized, images_13)

    # Convert to typical PyTorch input
    images_float = params.to_float(images_14) / 255

    # Sharpness (Converted to Blur temporarily)
    shp = params.get_enhanced_fparam(augmentations[4]) #params.get_fparam(-0.9) * fn.random.coin_flip(probability=augmentations_5) + 1 #params.get_enhanced_fparam(1.0)
    images_float_2 = params.blend(fn.gaussian_blur(images_float, window_size=3, device="gpu"), images_float, shp)
    images_float_3 = dalimath.clamp(images_float_2, 0.0, 1.0)

    #images_norm = params.normalize(images_float)
    #images_final = params.to_bchw(images_norm)
    images_final = fn.crop_mirror_normalize(images_float_3, device="gpu", mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616], output_layout="CHW")

    return images_final, labels


def apply_cifar_val(input, labels):

    images = input.gpu()
    labels = labels.gpu()
    
    # Convert to typical PyTorch input
    images_float = fn.cast(images, device="gpu", dtype=types.DALIDataType.FLOAT) / 255
    #images_norm = fn.normalize(images_float, device="gpu", axes=(0, 1), batch=True)
    images_final = fn.crop_mirror_normalize(images_float, device="gpu", mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616], output_layout="CHW")
    #images_final = fn.transpose(images_norm, device="gpu", perm=[2, 0, 1])

    return images_final, labels


def LCRAPipeline(iterator, batch_size, num_threads, device_id, magnitude, max_magnitude=10):
    pipe = Pipeline(batch_size, num_threads, device_id)
    params = RAOperations(device_id, magnitude, max_magnitude)

    # TODO: Potential issues with prefetch and setting particles

    with pipe:
        images, labels, augmentations = fn.external_source(iterator, num_outputs=3, layout=["HWC"]) # , cycle="raise")
        out_images, out_labels = apply_ra_single(images, labels, augmentations, params)
        pipe.set_outputs(out_images, out_labels)
    return pipe


def LCRAValPipeline(iterator, batch_size, num_threads, device_id):
    pipe = Pipeline(batch_size, num_threads, device_id)

    with pipe:
        images, labels = fn.external_source(iterator, num_outputs=2, layout=["HWC"])
        out_images, out_labels = apply_cifar_val(images, labels)
        pipe.set_outputs(out_images, out_labels)
    return pipe


def get_lcra_train_iterator(path, batch_size, num_threads, device_id, num_gpus, magnitude, shuffle=True, last_batch_policy=LastBatchPolicy.DROP, last_batch_padded=False, sublist=None, max_magnitude=10):
    iterator = LMDBClassIter(path, batch_size, device_id=device_id, num_gpus=num_gpus, shuffle=shuffle, last_batch_policy=last_batch_policy, last_batch_padded=last_batch_padded, sublist=sublist)
    pipeline = LCRAPipeline(iterator, batch_size, num_threads, device_id, magnitude, max_magnitude=max_magnitude)
    return pipeline, iterator

def get_lcra_val_iterator(path, batch_size, num_threads, device_id, num_gpus, sublist=None):
    iterator = LMDBClassIter(path, batch_size, device_id=device_id, num_gpus=num_gpus, shuffle=False, last_batch_policy=LastBatchPolicy.PARTIAL, last_batch_padded=True, sublist=sublist, output_augmentations=False)
    pipeline = LCRAValPipeline(iterator, batch_size, num_threads, device_id)
    return pipeline, iterator