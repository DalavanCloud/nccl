#!/bin/bash
#Caffe2
gpumodel=$1
mode=$2 #GROUP PARALLEL
resdir="results_dlfw"
path=$resdir/caffe2
mkdir -p $path
result="$path/$gpumodel.$mode"

# Environment Variables
export CAFFE2_ROOT=/home/nightly/install/caffe2
export PYTHONPATH=$PYTHONPATH:$CAFFE2_ROOT/build

if [ "$gpumodel" == "dgx1v" ]; then
  extra="--enable-tensor-core"
fi

NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=WARN srun -p $gpumodel --exclusive \
python $CAFFE2_ROOT/nvidia-examples/imagenet/train_resnet.py --train-lmdb /data/imagenet/train-lmdb-256x256 --skip-test --batch-size 512 --num-iterations 500 $extra --all-gpus --epochs 1 | \
tee /dev/stderr | awk -F'=' '/Epoch/ {if (lines++) sum += $NF} END {print sum/(lines-1)}' | \
tee -a $result.out
