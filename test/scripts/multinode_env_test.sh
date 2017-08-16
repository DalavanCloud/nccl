#!/bin/bash

test_env() {
gpumodel=$1
nnode=$2
ngpus=$3
env=$4
shift 4
vals=$@

op=all_reduce
timeout=2
nproc=$nnode

if [ "$SLURM" == "1" ]; then
  salloc_cmd="salloc -p $gpumodel -N $nnode -n $nproc -c $ngpus -t ${timeout} --exclusive "
else
  mpi_hosts="-host $gpumodel -oversubscribe "
  if [ "$MPI_HOME" == "" ]; then
    echo "Please specify MPI_HOME by: export MPI_HOME=/path/to/MPI"
    exit 1
  fi
  prefix="--prefix $MPI_HOME "
fi

resdir="results_env"
path=$resdir/$gpumodel
subpath=$path/$env
mkdir -p $subpath

for val in $vals ; do
  echo "Running test/perf/${op}_perf with [$env=$val] ..."
  result=$subpath/$val
  $salloc_cmd mpirun $prefix $mpi_hosts -x NCCL_DEBUG -x $env=$val -np $nproc test/perf/${op}_perf -g $ngpus -b 64 -e 128M -f 8 -w 1 -n 5 2>&1 | tee $result.out
done
}

gpumodel=$1
nnode=$2
maxgpu=$3

if [ "$maxgpu" == "" ]; then
  echo "Usage : $0 <gpumodel> <nnode> <maxgpu>"
  exit 1
fi

export NCCL_DEBUG=INFO

test_env $gpumodel $nnode $maxgpu NCCL_IB_DISABLE 0 1
test_env $gpumodel $nnode $maxgpu NCCL_IB_CUDA_SUPPORT 0 1
test_env $gpumodel $nnode $maxgpu NCCL_IB_TIMEOUT 14 4
test_env $gpumodel $nnode $maxgpu NCCL_IB_HCA mlx5 mlx5_0:1 ^mlx5_0:1
test_env $gpumodel $nnode $maxgpu NCCL_SOCKET_IFNAME eth ^ib
test_env $gpumodel $nnode $maxgpu NCCL_SOCKET_FAMILY AF_INET4 AF_INET6
test_env $gpumodel $nnode $maxgpu NCCL_NET_GDR_READ 0 1
