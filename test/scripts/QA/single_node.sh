#!/bin/bash
cd ~/sw/gpgpu/nccl/gitfusion/master/build/

gpumodel=$1
maxgpu=$2

../test/scripts/run_perf_graphs.sh $gpumodel $maxgpu
../test/scripts/run_perf_graphs.sh $gpumodel $maxgpu nocheck reorder
../test/scripts/run_perf_graphs.sh $gpumodel $maxgpu all
../test/scripts/run_perf_graphs.sh $gpumodel $maxgpu nocheck mpi
