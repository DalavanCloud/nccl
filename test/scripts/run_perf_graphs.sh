#!/bin/bash

generate_perf() {
gpumodel=$1
ngpus=$2
check=$3
mpi=$4
all=$5
reorder=$6
op=$7

timeout=2
extra="-c $check "
if [ "$all" == "1" ]; then
  # This is way too long already
  #extra+="-d all -o all "
  extra+="-d all "
  timeout=`expr $timeout \* 28`
fi

resdir="results"

if [ "$mpi" == "1" ]; then
  resdir+="_mpi"
fi

SET_VISIBLE=""
if [ "$reorder" == "1" ]; then
  GPU_REORDER=""
  for gpu in `seq $ngpus -1 0`; do
    GPU_REORDER+=",$gpu"
  done
  GPU_REORDER=`echo $GPU_REORDER | cut -c 4-`
  if [ "$mpi" == "1" ]; then
    SET_VISIBLE+="-x "
  fi
  SET_VISIBLE+="CUDA_VISIBLE_DEVICES=$GPU_REORDER"
  resdir+="_reorder"
fi

result=$resdir/$gpumodel/$op.$ngpus

mkdir -p $resdir/$gpumodel/
if [ "$mpi" == "0" ]; then
  echo "Running test/perf/${op}_perf on $ngpus GPUs ..."
  eval $SET_VISIBLE srun -p $gpumodel -n 1 -c $ngpus -t ${timeout} test/perf/${op}_perf -g $ngpus -b 40000 -e 1960000 -i 40000 $extra -w 20 -n 20 | tee $result.out
  eval $SET_VISIBLE srun -p $gpumodel -n 1 -c $ngpus -t ${timeout} test/perf/${op}_perf -t $ngpus -p 1 -b 2000000 -e 38000000 -i 2000000 $extra -w 20 -n 5 | tee -a $result.out
  eval $SET_VISIBLE srun -p $gpumodel -n 1 -c $ngpus -t ${timeout} test/perf/${op}_perf -t $ngpus -b 40000000 -e 400000000 -i 40000000 $extra -w 5 -n 1 | tee -a $result.out
else
  echo "Running test/perf/${op}_perf on $ngpus GPUs [MPI] ..."
  salloc -p $gpumodel -n $ngpus -c 1 -t ${timeout} mpirun -x NCCL_DEBUG $SET_VISIBLE -np $ngpus test/perf/${op}_perf -b 40000 -e 1960000 -i 40000 $extra -w 20 -n 20 | tee $result.out
  salloc -p $gpumodel -n $ngpus -c 1 -t ${timeout} mpirun -x NCCL_DEBUG $SET_VISIBLE -np $ngpus test/perf/${op}_perf -b 2000000 -e 38000000 -i 2000000 $extra -w 20 -n 5 | tee -a $result.out
  salloc -p $gpumodel -n $ngpus -c 1 -t ${timeout} mpirun -x NCCL_DEBUG $SET_VISIBLE -np $ngpus test/perf/${op}_perf -b 40000000 -e 400000000 -i 40000000 $extra -w 5 -n 1 | tee -a $result.out
fi
}

perf_ngpu_loop() {
gpumodel=$1
maxgpu=$2
check=$3
mpi=$4
all=$5
reorder=$6
op=$7
for ngpus in `seq 2 2 $maxgpu`; do
  generate_perf $gpumodel $ngpus $check $mpi $all $reorder $op 
done
}

gpumodel=$1
maxgpu=$2

if [ "$maxgpu" == "" ]; then
  echo "Usage : $0 <gpumodel> <maxgpus>"
  exit 1
fi

check=1
mpi=0
all=0
reorder=0
while [ "$3" != "" ]; do
  if [ "$3" == "nocheck" ]; then
    check=0
  fi
  if [ "$3" == "mpi" ]; then
    mpi=1
  fi
  if [ "$3" == "all" ]; then
    all=1
  fi
  if [ "$3" == "reorder" ]; then
    reorder=1
  fi
  shift
done

export NCCL_DEBUG=WARN

perf_ngpu_loop $gpumodel $maxgpu $check $mpi $all $reorder reduce
perf_ngpu_loop $gpumodel $maxgpu $check $mpi $all $reorder broadcast
perf_ngpu_loop $gpumodel $maxgpu $check $mpi $all $reorder all_reduce
perf_ngpu_loop $gpumodel $maxgpu $check $mpi $all $reorder all_gather
perf_ngpu_loop $gpumodel $maxgpu $check $mpi $all $reorder reduce_scatter
