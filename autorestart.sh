#!/bin/zsh
#freturn() { return "$1" ; } 
WDIR="$(dirname "$(readlink -f "$0")")"
cmd="python3 $WDIR/nnopt.py 4 filter.enabled=True dataset_params.num_classes=1000 val_dataset_path=~/datasets/ImageNet/val.lmdb train_dataset_path=~/datasets/ImageNet/train.lmdb ra_n=2 ra_m=4 model=resnet model_params.depth=50 lr=0.025 batch_size=128 dataset=imagenet epochs=180 batch_size_t_mult=1400 batch_size_v_mult=160 filter.interval=2 pval_threads=2 ra_n=2 ra_m=5 filter.init_val=1.0 filter.unit_vec_init=True filter.velocity=0.004"
#cmda="$cmd resume=/home/tsaregorodtsev/nnopt/imagenet_ckpt.pth.tar"
eval ${cmd}
#freturn 1
while [ $? -ne 0 ]; do
    dir=$(ls -d /home/tsaregorodtsev/21*/ | tail -n 1)
    file="${dir}model_best.pth.tar"
    newcmd="$cmd resume=$file"
    eval ${newcmd}
    #freturn 0
    #echo $?
    #echo $file
done
