#!/bin/bash

gpumodel=$1

maxgpu=$2

mode=$3

# get dir of test scripts
SHDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SHDIR/../../
NCCLROOT=$PWD
BLDDIR=$NCCLROOT/build
rm $BLDDIR/state

# DGX specific setting
if [ "$gpumodel" == "dgx1" ]; then
  module load cuda
  MPI_HOME="${MPI_HOME:-$HOME/install/openmpi}"
elif [ "$gpumodel" == "dgx1v" ]; then
  source $HOME/cuda.sh
  MPI_HOME="${MPI_HOME:-$HOME/install/openmpi}"
else
  source $SHDIR/cuda.sh
  MPI_HOME="${MPI_HOME:-/opt/mpi/openmpi}"
fi

# MPI Env
export MPI_HOME
export PATH=$MPI_HOME/bin:$PATH
if [ "$( which mpirun )" == "" ]; then
  echo "Cannot find MPI, please specify path using MPI_HOME=/path/to/MPI"
  exit 1
fi
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH

# build
if [ "$DEBDIR" == "" ] && [ "$INSTALL" != "1" ]; then
  make -j src.build
  DEBDIR=$BLDDIR
  export LD_LIBRARY_PATH=$DEBDIR/lib:$LD_LIBRARY_PATH
fi

if [ "$mode" == "dlfw" ] && [ "$gpumodel" == "P100" ]; then
  cd $BLDDIR
  $SHDIR/mxnet.sh $gpumodel
  $SHDIR/caffe2.sh $gpumodel
  #$SHDIR/tensorflow.sh $gpumodel
  #$SHDIR/pytorch.sh $gpumodel
  #$SHDIR/cntk.sh $gpumodel
elif [[ "$mode" == *"mpi"* ]] || [[ "$mode" == *"multinode"* ]]; then
  # test (multi processes)
  cd $NCCLROOT
  make -j test.clean
  if [ "$INSTALL" == "1" ]; then
    make -j test.build MPI=1
  else
    make -j test.build MPI=1 NCCLDIR=${DEBDIR}
  fi
  cd $BLDDIR
  if [[ "$mode" == *"mpi"* ]]; then
    echo "Testing $mode..."
    $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu $mode
  fi
  # multinode test
  if [[ "$mode" == *"multinode"* ]]; then
    if [ "$gpumodel" == "dgx1" ]; then
      $SHDIR/multinode_perf_graphs.sh dgx1 2 16 8 8
    elif [ "$gpumodel" == "P100" ]; then
      $SHDIR/multinode_perf_graphs.sh gpu-verbs 2 16 8 8
    else
      echo "No multi-node test on $gpumodel"
    fi
  fi
else
  # test (single process)
  cd $NCCLROOT
  make -j test.clean
  if [ "$INSTALL" == "1" ]; then
    make -j test.build
  else
    make -j test.build NCCLDIR=${DEBDIR}
  fi
  cd $BLDDIR
  $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu $mode
fi

echo "NCCL_Complete" > state
