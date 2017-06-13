#!/bin/bash

gpumodel=$1
prefix=${gpumodel:0:3}

# get dir of test scripts
SHDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SHDIR/../../
NCCLROOT=$PWD
BLDDIR=$NCCLROOT/build
rm $BLDDIR/state

# DGX specific setting
if [ "$prefix" == "DGX" ]; then
  module load cuda
  install_dir=$HOME/install
  SRUN="srun -p dgx1 -u "
  SALLOC="salloc -N1 -p dgx1 "
else
  source $SHDIR/cuda.sh
  install_dir=/mnt/linuxqa/$USER/install
fi

# build
make clean
make -j src.build

# test (single process)
make -j test.clean
make -j test.build
cd $BLDDIR
export LD_LIBRARY_PATH=$BLDDIR/lib:$LD_LIBRARY_PATH
$SRUN $SHDIR/run_perf_graphs.sh $gpumodel 8
$SRUN $SHDIR/run_perf_graphs.sh $gpumodel 8 nocheck reorder

# test (multi processes)
cd $NCCLROOT
make -j test.clean
lib=openmpi
export OPAL_PREFIX=$install_dir/$lib
export PATH=$OPAL_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$OPAL_PREFIX/lib:$LD_LIBRARY_PATH
export MPI_HOME=$OPAL_PREFIX
make -j test.build MPI=1
cd $BLDDIR
export LD_LIBRARY_PATH=$BLDDIR/lib:$LD_LIBRARY_PATH
$SALLOC $SHDIR/run_perf_graphs.sh $gpumodel 8 nocheck mpi
$SALLOC $SHDIR/run_perf_graphs.sh $gpumodel 8 nocheck mpi reorder

echo "NCCL_Complete" > state
