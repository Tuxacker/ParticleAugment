#!/bin/zsh
DIR="$(dirname "$(readlink -f "$0")")"
#python3 nnopt.py 2 filter.enabled=True
#python3 nnopt.py 2 filter.enabled=True lr=0.025
#python3 nnopt.py 2 filter.enabled=True ra_n=3 ra_m=3
#python3 nnopt.py 2 filter.enabled=True ra_tf=increasing
#python3 nnopt.py 2 filter.enabled=True ra_n=3 ra_m=1
#python3 nnopt.py 2 filter.enabled=True ra_n=3 ra_m=2
#python3 nnopt.py 2 filter.enabled=True ra_n=3 ra_m=3
#python3 nnopt.py 2 filter.enabled=True ra_n=2 ra_m=2
#python3 nnopt.py 2 filter.enabled=True ra_n=4 ra_m=2
#python3 nnopt.py 2 filter.enabled=True model_params.width=10 ra_n=3 ra_m=1
#python3 nnopt.py 2 filter.enabled=True model_params.width=10 ra_n=3 ra_m=2
#python3 nnopt.py 2 filter.enabled=True model_params.width=10 ra_n=3 ra_m=3
#python3 nnopt.py 2 filter.enabled=True model_params.width=10 ra_n=2 ra_m=2
#python3 nnopt.py 2 filter.enabled=True model_params.width=10 ra_n=4 ra_m=2
#python3 $DIR/nnopt.py 4 filter.enabled=True dataset_params.num_classes=1000 val_dataset_path=~/datasets/ImageNet/val.lmdb train_dataset_path=~/datasets/ImageNet/train.lmdb ra_n=2 ra_m=4 model=resnet model_params.depth=50 lr=0.05 batch_size=128 dataset=imagenet epochs=180 batch_size_t_mult=500 batch_size_v_mult=60 filter.interval=3 pval_threads=2 resume=/home/tsaregorodtsev/210504173830/model_best.pth.tar
#python3 $DIR/nnopt.py 2 filter.enabled=True dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=3 ra_m=1
#python3 $DIR/nnopt.py 4 filter.enabled=True dataset_params.num_classes=1000 val_dataset_path=~/datasets/ImageNet/val.lmdb train_dataset_path=~/datasets/ImageNet/train.lmdb ra_n=2 ra_m=4 model=resnet model_params.depth=50 lr=0.025 batch_size=128 dataset=imagenet epochs=180 batch_size_t_mult=1400 batch_size_v_mult=40 filter.interval=2 pval_threads=2 filter.n_particles=128 filter.lr=0.15
#python3 $DIR/nnopt.py 2 filter.enabled=True dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=3 ra_m=2
#python3 $DIR/nnopt.py 2 filter.enabled=True model_params.width=10 dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=3 ra_m=7
#python3 $DIR/nnopt.py 2 filter.enabled=True model_params.width=10 dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=3 ra_m=8
#python3 $DIR/nnopt.py 2 filter.enabled=True model_params.width=10 dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=4 ra_m=6

python3 $DIR/nnopt.py 2 filter.enabled=True ra_n=3 ra_m=3
python3 $DIR/nnopt.py 2 filter.enabled=True model_params.width=10 ra_n=4 ra_m=2
python3 $DIR/nnopt.py 2 filter.enabled=True dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=2 ra_m=2
python3 $DIR/nnopt.py 2 filter.enabled=True model_params.width=10 dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=4 ra_m=6 filter.velocity=-0.001 filter.unit_vec_init=True
python3 $DIR/nnopt.py 2 filter.enabled=True ra_n=3 ra_m=3 model=shake_shake model_params.depth=26 model_params.width=96 filter.lr=0.25 epochs=750 lr=0.015

