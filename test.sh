#!/bin/zsh
#python3 nnopt.py 2 filter.enabled=True
#python3 nnopt.py 2 filter.enabled=True lr=0.025
#python3 nnopt.py 2 filter.enabled=True ra_n=3 ra_m=3
#python3 nnopt.py 2 filter.enabled=True ra_tf=increasing
python3 nnopt.py 2 filter.enabled=True ra_n=3 ra_m=1
python3 nnopt.py 2 filter.enabled=True ra_n=3 ra_m=2
python3 nnopt.py 2 filter.enabled=True ra_n=3 ra_m=3
python3 nnopt.py 2 filter.enabled=True ra_n=2 ra_m=2
python3 nnopt.py 2 filter.enabled=True ra_n=4 ra_m=2
python3 nnopt.py 2 filter.enabled=True model_params.width=10 ra_n=3 ra_m=1
python3 nnopt.py 2 filter.enabled=True model_params.width=10 ra_n=3 ra_m=2
python3 nnopt.py 2 filter.enabled=True model_params.width=10 ra_n=3 ra_m=3
python3 nnopt.py 2 filter.enabled=True model_params.width=10 ra_n=2 ra_m=2
python3 nnopt.py 2 filter.enabled=True model_params.width=10 ra_n=4 ra_m=2
python3 nnopt.py 2 filter.enabled=True dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=1 ra_m=1
python3 nnopt.py 2 filter.enabled=True dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=1 ra_m=2
python3 nnopt.py 2 filter.enabled=True dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=1 ra_m=3
python3 nnopt.py 2 filter.enabled=True model_params.width=10 dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=2 ra_m=4
python3 nnopt.py 2 filter.enabled=True model_params.width=10 dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=2 ra_m=5
python3 nnopt.py 2 filter.enabled=True model_params.width=10 dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=2 ra_m=6
