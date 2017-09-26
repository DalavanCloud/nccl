#Caffe2
gpumodel=$1
resdir="results_dlfw"
mkdir -p $resdir
output="$resdir/tensorflow.txt"

# Environment Variables
export INSTALL=/home/nightly/install
export TMP_PKG_DIR=$INSTALL/tmp_tensorflow_pkg

# TF needs installing python packages (we do that locally)
python -m pip install $TMP_PKG_DIR/*whl --user
python -m pip install --upgrade $TMP_PKG_DIR/*whl --user

BENCH_DIR=$INSTALL/tf_benchmarks/scripts/tf_cnn_benchmarks

for opt in "False" "True"; do
  echo -n "--use_nccl=$opt : " | tee -a $output
  NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=WARN srun -p $gpumodel --exclusive \
    python $BENCH_DIR/tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 --batch_size=128 --model=resnet50 --variable_update=replicated --use_nccl=$opt | \
    awk '/total images/ {print $NF}' | \
    tee -a $output
done
