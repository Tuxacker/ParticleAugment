from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from torch import mean, min as tmin, max as tmax
from torch.cuda import ByteTensor, HalfTensor, FloatTensor

from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

from data.dali_ra import get_lcra_train_iterator, get_lcra_val_iterator
from utils.config_src import get_global_config

def test_create_pipeline():
    config = get_global_config()
    pipeline, _ = get_lcra_train_iterator(config.tests.lmdb_dataset_path, 512, 32, 0, 2, 5)
    pipeline.build()
    assert pipeline is not None

def test_pipeline_output_image():
    config = get_global_config()
    pipeline, iterator = get_lcra_train_iterator(config.tests.lmdb_dataset_path, 4, 2, 0, 2, 25)
    pipeline.build()
    iterator.particles = np.array([0,0,0,0,0,0,0,0,0,1], dtype=np.float32)
    test_loader = DALIClassificationIterator(pipeline)
    for elem in test_loader:
        input = elem[0]["data"]
        target = elem[0]["label"]
        print("Image batch shape:", input.shape, "Average", mean(input), "Min", tmin(input), "Max", tmax(input))
        print("Label batch shape:", target.shape)
        input = input.type(FloatTensor).cpu().permute([0, 2, 3, 1]).numpy()
        input = input / 4 + 0.5
        target = target.cpu().numpy()
        fig, axes = plt.subplots(2, 2, dpi=384, facecolor='w', edgecolor='k')
        for i in range(4):
            axes[i // 2, i % 2].imshow(input[i])
            axes[i // 2, i % 2].set_title(target[i])
        fig.savefig("test_pipeline_output_image.png")
        break
    assert pipeline is not None

def test_val_pipeline_output_image():
    config = get_global_config()
    pipeline, _ = get_lcra_val_iterator(config.val_dataset_path, 4, 2, 0, 1)
    pipeline.build()
    test_loader = DALIClassificationIterator(pipeline)
    for elem in test_loader:
        input = elem[0]["data"]
        target = elem[0]["label"]
        print("Image batch shape:", input.shape, "Average", mean(input), "Min", tmin(input), "Max", tmax(input))
        print("Label batch shape:", target.shape)
        input = input.type(FloatTensor).cpu().permute([0, 2, 3, 1]).numpy()
        input = input / 4 + 0.5
        target = target.cpu().numpy()
        fig, axes = plt.subplots(2, 2, dpi=384, facecolor='w', edgecolor='k')
        for i in range(4):
            axes[i // 2, i % 2].imshow(input[i])
            axes[i // 2, i % 2].set_title(target[i])
        fig.savefig("val_pipeline_output_image.png")
        break
    assert pipeline is not None

def test_reset():
    config = get_global_config()
    pipeline, iterator = get_lcra_train_iterator(config.tests.lmdb_dataset_path, 3, 1, 0, 1, 25, last_batch_policy=LastBatchPolicy.PARTIAL, last_batch_padded=True, sublist=list(range(10)))
    pipeline.build()
    test_loader = DALIClassificationIterator(pipeline, last_batch_policy=LastBatchPolicy.PARTIAL, dynamic_shape=True)
    for _ in range(5):
        for elem in test_loader:
            input = elem[0]["data"]
            target = elem[0]["label"]
        test_loader.reset()

def test_pipeline_iteration_config():
    config = get_global_config()
    pipeline, iterator = get_lcra_train_iterator(config.tests.lmdb_dataset_path, 3, 1, 0, 2, 25, last_batch_policy=LastBatchPolicy.PARTIAL, last_batch_padded=False, sublist=list(range(10)), shuffle=False)
    pipeline.build()
    test_loader = DALIClassificationIterator(pipeline, last_batch_policy=LastBatchPolicy.PARTIAL, dynamic_shape=True)
    iterator.epoch = 0
    total_len = 0
    for i, elem in enumerate(test_loader):
        input = elem[0]["data"]
        target = elem[0]["label"]
        total_len += input.size()[0]
        #print("Batch {}, length: {}, targets: {}, total length: {}".format(i, input.size()[0], target.cpu(), total_len))
    pipeline, _ = get_lcra_train_iterator(config.tests.lmdb_dataset_path, 3, 1, 1, 2, 25, last_batch_policy=LastBatchPolicy.PARTIAL, last_batch_padded=False, sublist=list(range(10)), shuffle=False)
    pipeline.build()
    test_loader = DALIClassificationIterator(pipeline, last_batch_policy=LastBatchPolicy.PARTIAL, dynamic_shape=True)
    iterator.epoch = 0
    total_len = 0
    for i, elem in enumerate(test_loader):
        input = elem[0]["data"]
        target = elem[0]["label"]
        total_len += input.size()[0]
        #print("Batch {}, length: {}, targets: {}, total length: {}".format(i, input.size()[0], target.cpu(), total_len))


def _not_test_pipeline_output_speed():
    config = get_global_config()
    pipeline, _ = get_lcra_train_iterator(config.tests.lmdb_dataset_path, 512, 16, 0, 2, 25)
    pipeline.build()
    test_loader = DALIClassificationIterator(pipeline)
    loaded_samples = 0
    start = timer()
    for elem in test_loader:
        input = elem[0]["data"]
        target = elem[0]["label"]
        loaded_samples += input.size()[0]
    elapsed = timer() - start
    print("Speed: {:.2f} img/s, Total: {} images in {:.2f} seconds".format(loaded_samples / elapsed, loaded_samples, elapsed))
    assert pipeline is not None