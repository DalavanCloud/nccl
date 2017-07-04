#!/bin/bash
cd ~/sw/gpgpu/nccl/gitfusion/master/build/

ptn=$1

../test/scripts/multinode_perf_graphs.sh $ptn 2 16 8 8
