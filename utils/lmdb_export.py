
import io
import pathlib
import pickle
import sys
    
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from tqdm import tqdm

from utils.config_src import get_global_config
from utils.path_utils import get_fq_fpath, get_parent_dir, is_dir, join_paths

def export_classification_dataset(path, dataset, name="lmdb_dataset", write_frequency=5000, compress=False, limit=None):
    data_loader = DataLoader(dataset, num_workers=32, collate_fn=lambda x: x, shuffle=False)

    lmdb_path = join_paths(path, "{}.lmdb".format(name), check=False)
    print("Loading dataset at {}".format(get_parent_dir(lmdb_path)))
    isdir = is_dir(lmdb_path)

    print("Writing LMDB dataset to {}".format(lmdb_path))
    db = lmdb.open(lmdb_path, subdir=isdir, map_size=2 ** 41, readonly=False, meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(tqdm(data_loader)):
        if limit is not None and idx == limit:
            break
        image, label = data[0]

        with io.BytesIO() as arr:
            if compress:
                if not type(image) == Image.Image:
                    image = Image.fromarray(image)
                image.save(arr, format="JPEG")
            else:
                np.save(arr, np.array(image, dtype=np.uint8))
            txn.put("{}".format(idx).encode('ascii'), pickle.dumps((arr.getvalue(), label)))

        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    #keys = ["{}".format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        #txn.put(b'__keys__', pickle.dumps(keys))
        if limit is None:
            txn.put(b'__len__', pickle.dumps(len(dataset)))
        else:
            txn.put(b'__len__', pickle.dumps(limit))

    print("Flushing database ...")
    db.sync()
    db.close()

if __name__ == "__main__":
    compress = False
    lmdb_dataset_dir = sys.argv[4]
    if sys.argv[1] == "cifar10":
        dataset = CIFAR10(lmdb_dataset_dir, train=sys.argv[2] == "train")
    elif sys.argv[1] == "cifar100":
        dataset = CIFAR100(lmdb_dataset_dir, train=sys.argv[2] == "train")
    elif sys.argv[1] == "imagenet":
        dataset = ImageNet(lmdb_dataset_dir, split=sys.argv[2])
        compress = True
    export_classification_dataset(lmdb_dataset_dir, dataset, name=sys.argv[3], compress=compress)
