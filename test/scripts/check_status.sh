#!/bin/bash
  
export TESTROOT=$HOME/nb-test
export NCCLROOT=$TESTROOT/nccl
cat $NCCLROOT/build/state 2>/dev/null
