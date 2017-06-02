#!/bin/bash

gpumodel=$1

# get dir of test scripts
SHDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SHDIR/../../
NCCLROOT=$PWD
BLDDIR=$NCCLROOT/build

# build
cd $NCCLROOT
source $SHDIR/cuda.sh
make -j src.build
make -j test.build

# test
cd $BLDDIR
LD_LIBRARY_PATH=$BLDDIR/lib:$LD_LIBRARY_PATH $SHDIR/run_perf_graphs.sh $gpumodel 8
echo "NCCL_Complete" > state
