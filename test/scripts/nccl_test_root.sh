#!/bin/bash

hostfile=$1
VER=$2
skiptest=0
TESTROOT=$HOME/nb-test

if [ "$hostfile" == "" ]; then
  hostfile=hostfile
fi

if [ "$VER" == "" ]; then
  VER=master
fi

if [ $skiptest -eq 0 ]; then
  while IFS= read -r var
  do
    arr=( $var )
    HOST=${arr[0]}
    GPUMODEL=${arr[1]}
    ssh $USER@$HOST gpumodel=$GPUMODEL TESTROOT=$TESTROOT VER=$VER 'bash -s' <<'ENDSSH'
    export NCCLROOT=$TESTROOT/nccl
    rm -rf $TESTROOT
    mkdir $TESTROOT; cd $TESTROOT
    hostname; pwd; echo $gpumodel
    git clone -b $VER ssh://kwen@git-master.nvidia.com:12001/cuda_ext/nccl.git
    cd nccl
    ./test/scripts/nccl_test_node.sh $gpumodel &
ENDSSH
  done < "$hostfile"
fi

# Gather results from test nodes
rm -rf $VER.bak
mv $VER $VER.bak
rm -rf $VER
mkdir $VER

while IFS= read -r var
do
  arr=( $var )
  HOST=${arr[0]}
  GPUMODEL=${arr[1]}
  time=0
  state=$( ssh $USER@$HOST < check_status.sh | grep 'NCCL_Complete' )
  while [ "$state" == "" ]
  do
    if [ $time -gt 300 ]; then
      echo $HOST "time out!"
      exit 1
    fi
    sleep 1m
    time=$(expr $time + 1)
    state=$( ssh $USER@$HOST < check_status.sh | grep 'NCCL_Complete' )
  done
  rsync -avzhe ssh $USER@$HOST:$TESTROOT/nccl/build/results/$GPUMODEL ./$VER/ 
  echo $HOST "completes"
  echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
  echo
done < "$hostfile"

echo "=========================="
echo "||  ALL TESTS COMPLETE  ||"
echo "=========================="
