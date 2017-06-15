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
else
  source $SHDIR/cuda.sh
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
  cd $NCCLROOT
  make -j test.clean
  install_dir=$HOME/install
  lib=openmpi
  export OPAL_PREFIX=$install_dir/$lib
  export PATH=$OPAL_PREFIX/bin:$PATH
  export LD_LIBRARY_PATH=$OPAL_PREFIX/lib:$LD_LIBRARY_PATH
  export MPI_HOME=$OPAL_PREFIX
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
