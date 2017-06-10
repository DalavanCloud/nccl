#!/bin/bash

gpumodel=$1

# get dir of test scripts
SHDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SHDIR/../../
NCCLROOT=$PWD
BLDDIR=$NCCLROOT/build
rm $BLDDIR/state

# build
source $SHDIR/cuda.sh
make clean
make -j src.build

# test (single process)
make -j test.clean
make -j test.build
cd $BLDDIR
LD_LIBRARY_PATH=$BLDDIR/lib:$LD_LIBRARY_PATH $SHDIR/run_perf_graphs.sh $gpumodel 8
LD_LIBRARY_PATH=$BLDDIR/lib:$LD_LIBRARY_PATH $SHDIR/run_perf_graphs.sh $gpumodel 8 nocheck reorder

# test (multi processes)
cd $NCCLROOT
make -j test.clean
install_dir=/mnt/linuxqa/kwen/install
lib=openmpi
export OPAL_PREFIX=$install_dir/$lib
export PATH=$OPAL_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$OPAL_PREFIX/lib:$LD_LIBRARY_PATH
export MPI_HOME=$OPAL_PREFIX
make -j test.build MPI=1
cd $BLDDIR
LD_LIBRARY_PATH=$BLDDIR/lib:$LD_LIBRARY_PATH $SHDIR/run_perf_graphs.sh $gpumodel 8 nocheck mpi

echo "NCCL_Complete" > state
