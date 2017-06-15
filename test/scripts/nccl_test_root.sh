#!/bin/bash

hostfile=$1
VER=$2
skiptest=0
TESTROOT=/home/$USER/nb-test
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
    printf "\nLaunching test on $HOST ...\n" 
    ssh $USER@$HOST -nt screen -dmS $SID "
    export NCCLROOT=$TESTROOT/nccl
    rm -rf $TESTROOT; mkdir $TESTROOT
    cd $TESTROOT
    hostname; pwd; echo $GPUMODEL
    git clone -b $VER ssh://kwen@git-master.nvidia.com:12001/cuda_ext/nccl.git
    cd nccl
    ./test/scripts/nccl_test_node.sh $GPUMODEL < /dev/null > $HOST.log 2>&1 &
    "
  done < "$hostfile"
fi

# Gather results from test nodes
rm -rf $VER.bak; rm -rf ${VER}_*.bak
mv $VER $VER.bak; mv ${VER}_mpi ${VER}_mpi.bak; mv ${VER}_reorder ${VER}_reorder.bak 

while IFS= read -r var
do
  arr=( $var )
  HOST=${arr[0]}
  GPUMODEL=${arr[1]}
  time=0
  state=""
  printf "\nWaiting for $HOST "
  while [ "$state" == "" ] && [ $time -lt 400 ]
  do
    printf "."
    sleep 1m
    time=$(expr $time + 1)
    state=$( ssh $USER@$HOST < check_status.sh 2>/dev/null | grep 'NCCL_Complete' )
  done
  if [ "$state" == "" ] && [ $time -eq 400 ]; then
    printf "\n$HOST TIME OUT!\n"
  else
    printf "\n$HOST COMPLETES\n"
  fi
  for opt in "" "_mpi" "_reorder"; do
    subVER=${VER}${opt}
    resdir="results$opt"
    mkdir -p $subVER/results
    rsync -avzhe ssh $USER@$HOST:$TESTROOT/nccl/build/$resdir/$GPUMODEL ./$subVER/results/
  done
  ssh $USER@$HOST -n "screen -X -S $SID quit"
  printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
done < "$hostfile"

echo "=========================="
echo "||  ALL TESTS COMPLETE  ||"
echo "=========================="
