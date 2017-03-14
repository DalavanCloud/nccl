/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GROUP_H_
#define NCCL_GROUP_H_

#include "nccl.h"
#include "core.h"

bool ncclAsyncMode();
ncclResult_t ncclAsyncErrCheck(ncclResult_t ret);

typedef ncclResult_t(*ncclInitFunc_t)(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank);

ncclResult_t ncclAsyncInit(ncclInitFunc_t func, int cudaDev, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank);

typedef ncclResult_t(*ncclCollFunc_t)(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclAsyncColl(ncclCollFunc_t func, const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

static ncclResult_t ncclCpuBarrierCheckin(ncclComm_t comm) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  int val = *ptr;
  bool done = false;
  // Reset other barrier if I'm the last
  while (done == false) {
    if (val+1 == comm->nRanks) {
      comm->intraBarrier[comm->intraPhase^1] = 0;
    }
    done = __sync_bool_compare_and_swap(ptr, val, val+1);
    val++;
  }
  return ncclSuccess;
}
static ncclResult_t ncclCpuBarrierWait(ncclComm_t comm) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  while (*ptr < comm->intraRanks) pthread_yield();
  comm->intraPhase ^= 1;
  return ncclSuccess;
}
#endif
