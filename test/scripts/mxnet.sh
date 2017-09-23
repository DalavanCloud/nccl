gpumodel=$1
resdir="results_dlfw"
mkdir -p $resdir
output="$resdir/mxnet.txt"

mxnet_path=/home/kwen/install/mxnet.redbcast

for opt in "nccl_reducebcast" "nccl_allreduce" "nccl" "device"; do
  echo "\noption = $opt : " | tee -a $output
  NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=VERSION srun -p $gpumodel --exclusive \
    python $mxnet_path/example/image-classification/train_imagenet.py --gpu 0,1,2,3,4,5,6,7 --batch-size 1024 --num-epochs 1 --disp-batches 100 --network resnet-v1 --num-layers 50 --dtype float16 --benchmark 1 --kv-store $opt 2>&1 | \
    awk '/Speed/ {if (lines++) sum += $5} END {print "average = " sum/(lines-1)}' | \
    grep "average = " | tee -a $output
done
