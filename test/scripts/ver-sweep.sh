#!/bin/bash

DATE=$(date --date="today" +%Y%m%d)
echo $DATE > $HOME/LASTLY_RUN

gpumodel=$1
maxgpu=$2
shift 2
VERS="$@"
OLD_VERS="1.5.4 1.6.5 2.0.2 2.0.5"

if [ "$VERS" == "" ]; then
  VERS="master"
fi

INSTVER=2.0.5
TESTDIR=$HOME/$DATE
NCCLDEB=$HOME/install/nccl.deb

export SLURM=1
OPTS="single latency reorder all mpi mpi_latency multinode dlfw"
MODES="GROUP PARALLEL"

for ver in $VERS ; do
   version_checked=$(echo $ver | sed -e '/^[0-9]*\.[0-9]*\.[0-9]*$/d')
   if [ -n "$version_checked" ]; then
     # is a branch
     git_prefix="-b $ver --single-branch"
   else
     # is a version
     git_prefix=""
   fi
   for mode in $MODES; do
     export NCCL_LAUNCH_MODE=$mode
     tag=$ver$mode
     ROOT=$TESTDIR/$tag-eris-$gpumodel
     rm -rf $ROOT; mkdir -p $ROOT; cd $ROOT
     git clone $git_prefix ssh://kwen@git-master.nvidia.com:12001/cuda_ext/nccl.git nccl.master
     git clone ssh://kwen@git-master.nvidia.com:12001/cuda_ext/nccl.git test.master
     rm -rf nccl.master/test
     mv test.master/test nccl.master/
     cd nccl.master/test/scripts
     for OPT in $OPTS; do
        if [ -n "$version_checked" ]; then
           ./nccl_test_node.sh $gpumodel $maxgpu $OPT $mode
        elif [ "$ver" == "$INSTVER" ]; then
           INSTALL=1 ./nccl_test_node.sh $gpumodel $maxgpu $OPT $mode
        #elif [ "$ver" == "1.5.4" ]; then
        #   NCCL_TOPOLOGY=CUBEMESH DEBDIR=$NCCLDEB/$ver ./nccl_test_node.sh $gpumodel $maxgpu $OPT
        else
           DEBDIR=$NCCLDEB/$ver ./nccl_test_node.sh $gpumodel $maxgpu $OPT $mode
        fi
     done
   done
done

transfer=1
if [ "$transfer" == "1" ]; then
  cd $HOME/install/nccl.deb/
  for VER in $VERS ; do
    for MODE in $MODES; do
       for OPT in "" "_mpi" "_reorder" "_all" "_multinode" "_latency" "_mpi_latency" "_multinode_latency"; do
          TAG=$VER$MODE
          DIR=$TAG$OPT
          echo "Syncing $DIR"
          rsync --update -ra $TESTDIR/$TAG-eris-$gpumodel/nccl.master/build/results$OPT/* $DIR/results/
       done
     done
  done
fi
