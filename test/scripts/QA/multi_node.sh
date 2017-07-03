#!/bin/bash
cd ~/sw/gpgpu/nccl/gitfusion/stable/build/

ptn=$1

../test/scripts/multinode_perf_graphs.sh $ptn 2 16 8 8
