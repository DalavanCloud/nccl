#Caffe2
gpumodel=$1
resdir="results_dlfw"
mkdir -p $resdir
output="$resdir/caffe2.txt"

# Environment Variables
export CAFFE2_ROOT=/home/nightly/install/caffe2
export PYTHONPATH=$PYTHONPATH:$CAFFE2_ROOT/build

if [ "$gpumodel" == "dgx1v" ]; then
  extra="--enable-tensor-core"
fi

for opt in "" "--use_nccl"; do
  echo -e "\noption = $opt : " | tee -a $output
  NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=WARN srun -p $gpumodel --exclusive \
    python $CAFFE2_ROOT/nvidia-examples/imagenet/train_resnet.py --synthetic-data --batch-size 1024 --dtype float16 $extra --all-gpus --workers-per-gpu 4 --epochs 1 | \
    tee /dev/stderr | awk -F'=' '/Epoch/ {if (lines++) sum += $NF} END {print "average = " sum/(lines-1)}' | \
    grep "average = " | tee -a $output
done
