
import io
import pickle

import lmdb
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from utils.config_src import get_global_config
from utils.path_utils import get_fq_fpath, get_parent_dir, is_dir, join_paths

def export_classification_dataset(path, dataset, name="lmdb_dataset", write_frequency=5000):
    data_loader = DataLoader(dataset, num_workers=32, collate_fn=lambda x: x, shuffle=False)

    lmdb_path = join_paths(path, "{}.lmdb".format(name), check=False)
    print("Loading dataset at {}".format(get_parent_dir(lmdb_path)))
    isdir = is_dir(lmdb_path)

    print("Writing LMDB dataset to {}".format(lmdb_path))
    db = lmdb.open(lmdb_path, subdir=isdir, map_size=2 ** 41, readonly=False, meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(tqdm(data_loader)):
        image, label = data[0]

        with io.BytesIO() as arr:
            np.save(arr, np.array(image, dtype=np.uint8))
            txn.put("{}".format(idx).encode('ascii'), pickle.dumps((arr.getvalue(), label)))

        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    #keys = ["{}".format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        #txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(dataset)))

    print("Flushing database ...")
    db.sync()
    db.close()

if __name__ == "__main__":

    import sys
    
    sys.path.append(get_parent_dir(__file__, go_back=1))

    config = get_global_config()
    dataset = CIFAR10(config.lmdb_export.lmdb_dataset_dir, train=True)
    export_classification_dataset(config.lmdb_export.lmdb_dataset_dir, dataset)
