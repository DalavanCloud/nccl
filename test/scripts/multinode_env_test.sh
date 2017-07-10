#!/bin/bash

gpumodel=$1
nnode=$2
nproc=$3
nthread=$4
ngpus=$5
op=all_reduce

if [ "$ngpus" == "" ]; then
  echo "Usage : $0 <gpumodel> <nnode> <maxproc> <maxthread> <maxgpu>"
  exit 1
fi

export NCCL_DEBUG=INFO

resdir="results_multinode"

timeout=3
extra="-c 0 "

mkdir -p $resdir/$gpumodel/

result=$resdir/$gpumodel/env_test

if [ "$SLURM" == "1" ]; then
  salloc_cmd="salloc -p $gpumodel -N $nnode -n $nproc -t ${timeout} "
else
  mpi_hosts="-host $gpumodel "
  if [ "$MPI_HOME" == "" ]; then
    echo "Please specify MPI_HOME by: export MPI_HOME=/path/to/MPI"
    exit 1
  fi
  prefix="--prefix $MPI_HOME "
fi

npn=$(expr $nproc / $nnode)

$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 80000 -i 40000 $extra -w 1 -n 1 2>&1 | tee $result.out
$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -x NCCL_IB_HCA=mlx5 -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 80000 -i 40000 $extra -w 1 -n 1 2>&1 | tee -a $result.out
$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -x NCCL_IB_HCA=mlx5_0:1,mlx5_1:1 -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 80000 -i 40000 $extra -w 1 -n 1 2>&1 | tee -a $result.out
$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -x NCCL_IB_HCA=^mlx5_0:1 -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 80000 -i 40000 $extra -w 1 -n 1 2>&1 | tee -a $result.out
$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -x NCCL_IB_HCA=^mlx5 -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 80000 -i 40000 $extra -w 1 -n 1 2>&1 | tee -a $result.out
$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -x NCCL_SOCKET_IFNAME=eth -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 80000 -i 40000 $extra -w 1 -n 1 2>&1 | tee -a $result.out
$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -x NCCL_SOCKET_IFNAME=^ib -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 80000 -i 40000 $extra -w 1 -n 1 2>&1 | tee -a $result.out
$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -x NCCL_IB_DISABLE=1 -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 80000 -i 40000 $extra -w 1 -n 1 2>&1 | tee -a $result.out
$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -x NCCL_IB_DISABLE=1 -x NCCL_SOCKET_IFNAME=eth1 -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 80000 -i 40000 $extra -w 1 -n 1 2>&1 | tee -a $result.out
$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -x NCCL_IB_DISABLE=1 -x NCCL_SOCKET_FAMILY=AF_INET6 -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 80000 -i 40000 $extra -w 1 -n 1 2>&1 | tee -a $result.out
$salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -x NCCL_IB_DISABLE=1 -x NCCL_SOCKET_IFNAME=^ib -x NCCL_SOCKET_FAMILY=AF_INET6 -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 80000 -i 40000 $extra -w 1 -n 1 2>&1 | tee -a $result.out
