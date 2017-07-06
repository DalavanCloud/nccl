#!/bin/bash
cd ../../../build/

ptn=$1

../test/scripts/multinode_perf_graphs.sh $ptn 2 16 8 8
