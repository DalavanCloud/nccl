#!/bin/bash

hostfile=$1
VER=$2
skiptest=0
TESTROOT=$HOME/nb-test
SID=$(date +%Y%m%d%H%M%S)
printf "\nScreen session ID is $SID (on all machines) \n\n"

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
    ssh $USER@$HOST -t screen -S $SID -m -d "
    export NCCLROOT=$TESTROOT/nccl
    rm -rf $TESTROOT; mkdir $TESTROOT 
    cd $TESTROOT
    hostname; pwd; echo $GPUMODEL
    git clone -b $VER ssh://kwen@git-master.nvidia.com:12001/cuda_ext/nccl.git
    cd nccl
    ./test/scripts/nccl_test_node.sh $GPUMODEL &"
  done < "$hostfile"
fi

# Gather results from test nodes
rm -rf $VER.bak
mv $VER $VER.bak
rm -rf $VER
mkdir -p $VER/results

while IFS= read -r var
do
  arr=( $var )
  HOST=${arr[0]}
  GPUMODEL=${arr[1]}
  time=0
  state=""
  printf "\nWaiting for $HOST "
  while [ "$state" == "" ]
  do
    printf "."
    if [ $time -gt 300 ]; then
      echo $HOST "time out!"
      exit 1
    fi
    sleep 1m
    time=$(expr $time + 1)
    state=$( ssh $USER@$HOST < check_status.sh 2>/dev/null | grep 'NCCL_Complete' )
  done
  echo
  rsync -avzhe ssh $USER@$HOST:$TESTROOT/nccl/build/results/$GPUMODEL ./$VER/results/
  ssh $USER@$HOST "screen -X -S $SID quit"
  printf "\n$HOST COMPLETES\n"
  printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
done < "$hostfile"

echo "=========================="
echo "||  ALL TESTS COMPLETE  ||"
echo "=========================="
