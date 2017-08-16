#!/bin/bash

test_env() {
gpumodel=$1
ngpus=$2
env=$3
shift 3
vals=$@

op=all_reduce
timeout=2

if [ "$SLURM" == "1" ]; then
  srun_cmd="srun -p $gpumodel -n 1 -c $ngpus -t ${timeout} --exclusive "
else
  srun_cmd="timeout ${timeout}m "
fi

resdir="results_env"
path=$resdir/$gpumodel
subpath=$path/$env
mkdir -p $subpath

for val in $vals ; do
  echo "Running test/perf/${op}_perf with [$env=$val] ..."
  result=$subpath/$val
  eval $env=$val $srun_cmd test/perf/${op}_perf -g $ngpus -b 64 -e 128M -f 8 -w 1 -n 5 2>&1 | tee $result.out
done
}

gpumodel=$1
maxgpu=$2

if [ "$maxgpu" == "" ]; then
  echo "Usage : $0 <gpumodel> <maxgpus>"
  exit 1
fi

export NCCL_DEBUG=INFO

negring=""
for gpu in `seq $maxgpu -1 0`; do
  negring+=",$gpu"
done
negring=`echo $negring | cut -c 4-`
posring=`echo $negring | rev`
rings="$negring\|$posring"

test_env $gpumodel $maxgpu NCCL_MAX_NRINGS 1 4 6 8 12
test_env $gpumodel $maxgpu NCCL_NTHREADS 128 256 512
test_env $gpumodel $maxgpu NCCL_RINGS $rings
test_env $gpumodel $maxgpu NCCL_SHM_DISABLE 0 1
test_env $gpumodel $maxgpu NCCL_NTHREADS 128 256 512
test_env $gpumodel $maxgpu NCCL_BUFFSIZE 1048576 4194304 8388608
