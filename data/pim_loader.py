import io
import os
import pickle

import lmdb
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit as Split

from .pim_ra import RandAugment, rand_augment_ops
from .pim_erasing import RandomErasing

class ClassLMDB(Dataset):

    def __init__(self, path, transform=None, subset=None):
        self.path = os.path.expanduser(path)
        self.env = lmdb.open(self.path, subdir=os.path.isdir(self.path), readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
        self.transform = transform
        self.subset = subset

    def __getstate__(self):
        state = self.__dict__
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.env = lmdb.open(self.path, subdir=os.path.isdir(self.path), readonly=True, lock=False, readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.subset is not None:
            index = self.subset[index]
        with self.env.begin(write=False) as txn:
            byteflow = txn.get("{}".format(index).encode("ascii"))
        image, label = pickle.loads(byteflow)

        with io.BytesIO() as arr:
            arr.write(image)
            arr.seek(0)
            image = np.load(arr, allow_pickle=True)
            image = Image.fromarray(image).convert('RGB')


        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.length if self.subset is None else len(self.subset)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.path + ')'

    def get_all_labels(self, save=False, path=None):
        if self.subset is not None:
            raise ValueError("Trying to save subset labels is not recommended")
        labels = np.empty((self.length), dtype=np.int64)
        with self.env.begin(write=False) as txn:
            for i in range(self.length):
                byteflow = txn.get("{}".format(i).encode("ascii"))
                _, label = pickle.loads(byteflow)
                labels[i] = label
        if save and path is not None:
            np.save(path, labels)
        return labels

    def get_balanced_subset(self, n_samples, seed=0):
        split = Split(n_splits=1, train_size=n_samples/len(self), random_state=seed)
        indices, _ = next(split.split(np.zeros((len(self))), self.get_all_labels()))
        return indices



def get_cifar_10_train_tf(magnitude):
    ops = rand_augment_ops(magnitude=magnitude)
    ra_instance = RandAugment(ops, 2)
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        ra_instance,
        transforms.ToTensor(),
        RandomErasing(device="cpu"),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]), ra_instance

def get_cifar_10_val_tf():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])


def get_cifar_10_train_loader(path, batch_size, num_threads, device_id, num_gpus, magnitude, subset=None, seed=0):
    transform, ra_instance = get_cifar_10_train_tf(magnitude)
    dataset = ClassLMDB(path, transform, subset=subset)
    sampler = DistributedSampler(dataset, num_replicas=num_gpus, rank=device_id, seed=seed) if num_gpus > 1 else None
    shuffle = None if num_gpus > 1 else True
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_threads, pin_memory=True), ra_instance

def get_cifar_10_val_loader(path, batch_size, num_threads, device_id, num_gpus):
    transform = get_cifar_10_val_tf()
    dataset = ClassLMDB(path, transform)
    sampler = DistributedSampler(dataset, num_replicas=num_gpus, rank=device_id, shuffle=False) if num_gpus > 1 else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=num_threads, pin_memory=True)
