#!/bin/bash

gpumodel=$1
prefix=${gpumodel:0:3}

maxgpu=$2

mpi=0
reorder=0
while [ "$3" != "" ]; do
  if [ "$3" == "mpi" ]; then
    mpi=1
  fi
  if [ "$3" == "reorder" ]; then
    reorder=1
  fi
  shift
done

# get dir of test scripts
SHDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SHDIR/../../
NCCLROOT=$PWD
BLDDIR=$NCCLROOT/build
rm $BLDDIR/state

# DGX specific setting
if [ "$prefix" == "dgx" ]; then
  module load cuda
  MPI_HOME="${MPI_HOME:-$HOME/install/openmpi}"
else
  source $SHDIR/cuda.sh
  MPI_HOME="${MPI_HOME:-/opt/mpi/openmpi}"
fi

# build
make -j src.build

if [ "$mpi" == "0" ]; then
  # test (single process)
  make -j test.clean
  make -j test.build
  cd $BLDDIR
  if [ "$reorder" == "0" ]; then
    echo "Tesing ..."
    LD_LIBRARY_PATH=$BLDDIR/lib:$LD_LIBRARY_PATH $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu
  else
    echo "Testing reorder..."
    LD_LIBRARY_PATH=$BLDDIR/lib:$LD_LIBRARY_PATH $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu nocheck reorder
  fi
fi

# test (multi processes)
if [ "$mpi" == "1" ]; then
  export PATH=$MPI_HOME/bin:$PATH
  if [ "$( which mpirun )" == "" ]; then
    echo "Cannot find MPI, please specify path using MPI_HOME=/path/to/MPI"
    exit 1
  fi
  export MPI_HOME
  export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
  cd $NCCLROOT
  make -j test.clean
  make -j test.build MPI=1
  cd $BLDDIR
  if [ "$reorder" == "0" ]; then
    echo "Testing MPI..."
    LD_LIBRARY_PATH=$BLDDIR/lib:$LD_LIBRARY_PATH $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu nocheck mpi
  else
    echo "Testing MPI+reorder..."
    LD_LIBRARY_PATH=$BLDDIR/lib:$LD_LIBRARY_PATH $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu nocheck mpi reorder
  fi
fi

echo "NCCL_Complete" > state
