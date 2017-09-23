#Caffe2
gpumodel=$1
resdir="results_dlfw"
mkdir -p $resdir
output="$resdir/cntk.txt"

BENCH_DIR=/home/nightly/install/cntk_tests

for opt in "" "-no-nccl"; do
  echo -e "\nCNTK$opt : " | tee -a $output

  # Environment Variables
  export CNTK_DIR=/home/nightly/install/cntk$opt
  export PATH=$CNTK_DIR/build/release/bin:$PATH
  export LD_LIBRARY_PATH=$CNTK_DIR/build/release/lib:$LD_LIBRARY_PATH

  NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=WARN salloc -p $gpumodel -n 8 \
    $BENCH_DIR/resnet50_cntk.sh | \
    tee /dev/stderr | awk '/^ Epoch/ {if (lines++ > 50) sum += $NF} END {print "average = " sum/(lines-1)}' | \
    grep "average = " | tee -a $output
done
