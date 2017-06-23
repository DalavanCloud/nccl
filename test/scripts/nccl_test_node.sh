#!/bin/bash

gpumodel=$1

maxgpu=$2

mpi=0
while [ "$3" != "" ]; do
  if [ "$3" == "mpi" ]; then
    mpi=1
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

# build
if [ "$DEBDIR" == "" ]; then
  make -j src.build
  DEBDIR=$BLDDIR
fi

if [ "$mpi" == "0" ]; then
  # test (single process)
  make -j test.clean
  make -j test.build NCCLDIR=${DEBDIR}
  cd $BLDDIR
  LD_LIBRARY_PATH=$DEBDIR/lib:$LD_LIBRARY_PATH $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu
  LD_LIBRARY_PATH=$DEBDIR/lib:$LD_LIBRARY_PATH $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu nocheck reorder
  LD_LIBRARY_PATH=$DEBDIR/lib:$LD_LIBRARY_PATH $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu all
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
  make -j test.build MPI=1 NCCLDIR=${DEBDIR}
  cd $BLDDIR
  echo "Testing MPI..."
  LD_LIBRARY_PATH=$DEBDIR/lib:$LD_LIBRARY_PATH $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu nocheck mpi
fi

echo "NCCL_Complete" > state
