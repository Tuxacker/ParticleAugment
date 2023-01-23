#!/bin/bash
singularity exec ~/nnopt.simg python3 lmdb_export.py imagenet val val /home/tsaregorodtsev/datasets/ImageNet/
