#!/bin/zsh
#freturn() { return "$1" ; } 
WDIR="$(dirname "$(readlink -f "$0")")"
cmd="nnopt.py 2 filter.enabled=True model_params.width=10 dataset_params.num_classes=100 val_dataset_path=~/datasets/CIFAR/cifar100-val.lmdb train_dataset_path=~/datasets/CIFAR/cifar100-train.lmdb ra_n=5 ra_m=6 epochs=200 lr=0.05 filter.n_particles=80"
cmd="python3 $WDIR/$cmd"
eval ${cmd}
while [[ $? -ne 0 ]]; do
    dir=$(cd $WDIR; ls -d 21*/ | tail -n 1)
    file="$WDIR/${dir}model_best.pth.tar"
    newcmd="$cmd resume=$file"
    eval ${newcmd}
done