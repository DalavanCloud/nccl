#PyTorch
gpumodel=$1
resdir="results_dlfw"
mkdir -p $resdir
output="$resdir/pytorch.txt"

# Environment Variables
export INSTALL=/home/nightly/install
export PATH=$INSTALL/conda/envs/pytorch-py35/bin:$PATH

BENCH_DIR=$INSTALL/pytorch/examples/imagenet

NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=WARN srun -p $gpumodel --exclusive \
  python $BENCH_DIR/main.py -a resnet50 /data/imagenet
