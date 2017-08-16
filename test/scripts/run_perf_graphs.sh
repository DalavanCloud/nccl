#!/bin/bash

generate_perf() {
gpumodel=$1
ngpus=$2
mode=$3
op=$4

resdir="results"
if [ "$mode" != "single" ]; then
  resdir+="_$mode"
fi
path=$resdir/$gpumodel
mkdir -p $path

timeout=2
extra="-c 1 "
if [ "$mode" == "all" ]; then
  timeout=`expr $timeout \* 28`
fi

if [ "$SLURM" == "1" ]; then
  srun_cmd="srun -p $gpumodel -n 1 -c $ngpus -t ${timeout} --exclusive "
  salloc_cmd="salloc -p $gpumodel -n $ngpus -c 1 -t ${timeout} --exclusive "
else
  srun_cmd="timeout ${timeout}m "
  salloc_cmd="timeout ${timeout}m "
  mpi_hosts="-host $gpumodel "
fi

if [ "$mode" == "reorder" ]; then
  echo "Running test/perf/${op}_perf on $ngpus GPUs [Reorder] ..."
  GPU_REORDER=""
  for gpu in `seq $ngpus -1 0`; do
    GPU_REORDER+=",$gpu"
  done
  GPU_REORDER=`echo $GPU_REORDER | cut -c 4-`
  result=$path/$op.$ngpus
  CUDA_VISIBLE_DEVICES=$GPU_REORDER $srun_cmd test/perf/${op}_perf -g $ngpus -b 40000000 -e 400000000 -i 40000000 -w 1 -n 5 2>&1 | tee -a $result.out
  return 0
fi

if [ "$mode" == "all" ]; then
  mkdir -p $path/pow2
  mkdir -p $path/npow2
  for dtype in float double half int8 int32 int64 uint8 uint32 uint64 ; do
    for otype in sum max min prod ; do
      echo "Running test/perf/${op}_perf on $ngpus GPUs [$dtype x $otype] ..."
      result=$op.$ngpus.$dtype.$otype
      $srun_cmd test/perf/${op}_perf -t $ngpus -d $dtype -o $otype -b 64 -e 4194304 -f 256 -w 1 -n 5 2>&1 | tee $path/pow2/$result.out
      $srun_cmd test/perf/${op}_perf -t $ngpus -d $dtype -o $otype -b 63 -e 4357647 -f 263 -w 1 -n 5 2>&1 | tee $path/npow2/$result.out
    done
  done
  return 0
fi

if [ "$mode" == "single" ]; then
  echo "Running test/perf/${op}_perf on $ngpus GPUs ..."
  result=$path/$op.$ngpus
  $srun_cmd test/perf/${op}_perf -t $ngpus -b 40000 -e 1960000 -i 40000 $extra -w 20 -n 20 2>&1 | tee $result.out
  $srun_cmd test/perf/${op}_perf -t $ngpus -p 1 -b 2000000 -e 38000000 -i 2000000 $extra -w 20 -n 5 2>&1 | tee -a $result.out
  $srun_cmd test/perf/${op}_perf -g $ngpus -b 40000000 -e 400000000 -i 40000000 $extra -w 5 -n 1 2>&1 | tee -a $result.out
  return 0
fi

if [ "$mode" == "mpi" ]; then
  echo "Running test/perf/${op}_perf on $ngpus GPUs [MPI] ..."
  result=$path/$op.$ngpus
  $salloc_cmd mpirun $mpi_hosts -x NCCL_DEBUG -np $ngpus test/perf/${op}_perf -b 40000 -e 1960000 -i 40000 -w 20 -n 20 2>&1 | tee $result.out
  $salloc_cmd mpirun $mpi_hosts -x NCCL_DEBUG -np $ngpus test/perf/${op}_perf -b 2000000 -e 38000000 -i 2000000  -w 20 -n 5 2>&1 | tee -a $result.out
  $salloc_cmd mpirun $mpi_hosts -x NCCL_DEBUG -np $ngpus test/perf/${op}_perf -b 40000000 -e 400000000 -i 40000000 -w 5 -n 1 2>&1 | tee -a $result.out
  return 0
fi
}

perf_ngpu_loop() {
gpumodel=$1
maxgpu=$2
mode=$3
op=$4
for ngpus in `seq 2 2 $maxgpu`; do
  generate_perf $gpumodel $ngpus $mode $op 
done
}

gpumodel=$1
maxgpu=$2

if [ "$maxgpu" == "" ]; then
  echo "Usage : $0 <gpumodel> <maxgpus>"
  exit 1
fi

mode=$3
if [ "$mode" == "" ]; then
  mode="single"
fi

export NCCL_DEBUG=WARN

if [ "$mode" == "reorder" ]; then
  perf_ngpu_loop $gpumodel $maxgpu $mode all_reduce
else
  perf_ngpu_loop $gpumodel $maxgpu $mode reduce
  perf_ngpu_loop $gpumodel $maxgpu $mode broadcast
  perf_ngpu_loop $gpumodel $maxgpu $mode all_reduce
  perf_ngpu_loop $gpumodel $maxgpu $mode all_gather
  perf_ngpu_loop $gpumodel $maxgpu $mode reduce_scatter
fi
