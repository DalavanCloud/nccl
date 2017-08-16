#!/bin/bash

generate_perf() {
gpumodel=$1
nnode=$2
nproc=$3
nthread=$4
ngpus=$5
op=$6

resdir="results_multinode"

timeout=3
extra="-c 0 "

mkdir -p $resdir/$gpumodel/

result=$resdir/$gpumodel/$op.$nproc.$nthread.$ngpus

nperslot=$(expr $nthread \* $ngpus)

if [ "$SLURM" == "1" ]; then
  salloc_cmd="salloc -p $gpumodel -N $nnode -n $nproc -c $nperslot -t ${timeout} --exclusive "
else
  mpi_hosts="-host $gpumodel -oversubscribe "
  if [ "$MPI_HOME" == "" ]; then
    echo "Please specify MPI_HOME by: export MPI_HOME=/path/to/MPI"
    exit 1
  fi
  prefix="--prefix $MPI_HOME "
fi

npn=$(expr $nproc / $nnode)

$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 1960000 -i 40000 $extra -w 5 -n 20 2>&1 | tee $result.out
$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 2000000 -e 38000000 -i 2000000 $extra -w 5 -n 5 2>&1 | tee -a $result.out
$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000000 -e 400000000 -i 40000000 $extra -w 1 -n 2 2>&1 | tee -a $result.out
}

perf_ptg_loop() {
gpumodel=$1
nnode=$2
maxproc=$3
maxthread=$4
maxgpu=$5
op=$6

totgpus=$(expr $nnode \* $maxgpu)

declare -i nproc=2
while [[ $nproc -le $maxproc ]] ; do
  declare -i nthread=1
  while [[ $nthread -le $maxthread ]] ; do
    declare -i ngpus=1
    while [[ $ngpus -le $maxgpu ]]; do
      ngperproc=$(expr $nthread \* $ngpus)
      if (( $( expr $nproc \* $ngperproc ) <= $totgpus )); then
        echo "Running test/perf/${op}_perf on $nnode nodes, $nproc processes, each process having $nthread threads with $ngpus GPUs ..."
        generate_perf $gpumodel $nnode $nproc $nthread $ngpus $op
      fi
      ngpus+=$ngpus
    done
    nthread+=$nthread
  done
  nproc+=$nproc
done
}

gpumodel=$1
nnode=$2
maxproc=$3
maxthread=$4
maxgpu=$5

if [ "$maxgpu" == "" ]; then
  echo "Usage : $0 <gpumodel> <nnode> <maxproc> <maxthread> <maxgpu>"
  exit 1
fi

export NCCL_DEBUG=WARN

perf_ptg_loop $gpumodel $nnode $maxproc $maxthread $maxgpu reduce
perf_ptg_loop $gpumodel $nnode $maxproc $maxthread $maxgpu all_reduce
perf_ptg_loop $gpumodel $nnode $maxproc $maxthread $maxgpu reduce_scatter
perf_ptg_loop $gpumodel $nnode $maxproc $maxthread $maxgpu all_gather
perf_ptg_loop $gpumodel $nnode $maxproc $maxthread $maxgpu broadcast
