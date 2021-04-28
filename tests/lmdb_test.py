import numpy as np
import matplotlib.pyplot as plt

from data.pim_loader import ClassLMDB

def test_lmdb():
    imagenet_lmdb = ClassLMDB("/share/Projects/Datasets/ImageNet/imagenet_test.lmdb", compressed=True)
    fig, axes = plt.subplots(2, 2, dpi=384, facecolor='w', edgecolor='k')
    for i in range(4):
        input, target = imagenet_lmdb[i]
        axes[i // 2, i % 2].imshow(input)
        axes[i // 2, i % 2].set_title(target)
    fig.savefig("test_lmdb.png")