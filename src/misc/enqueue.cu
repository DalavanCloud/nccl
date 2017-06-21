/*************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "common_coll.h"

ncclResult_t ncclCpuBarrierCheckin(ncclComm_t comm) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  int val = *ptr;
  bool done = false;
  // Reset other barrier if I'm the last
  while (done == false) {
    if (val >= comm->intraRanks) {
      WARN("Trying to launch too many collectives");
      return ncclInvalidUsage;
    }
    if (val+1 == comm->intraRanks) {
      comm->intraBarrier[comm->intraPhase^1] = 0;
    }
    done = __sync_bool_compare_and_swap(ptr, val, val+1);
    val++;
  }
  return ncclSuccess;
}
ncclResult_t ncclCpuBarrierWait(ncclComm_t comm) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  while (*ptr < comm->intraRanks) pthread_yield();
  comm->intraPhase ^= 1;
  return ncclSuccess;
}

ncclResult_t ncclEnqueueCheck(ncclFunc_t func, const char* primName, const void* sendbuff, 
    void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root,
    ncclComm_t comm, cudaStream_t stream) {
  // Launch asynchronously if needed
  if (ncclAsyncMode()) {
    int savedDev;
    CUDACHECK(cudaGetDevice(&savedDev));
    CUDACHECK(cudaSetDevice(comm->cudaDev));
    // Check arguments
    ncclResult_t ret = ArgsCheck(sendbuff, recvbuff, count, type, op, root, comm, primName);
    NCCLCHECK(ncclAsyncErrCheck(ret));
    NCCLCHECK(ncclAsyncColl(func, sendbuff, recvbuff, count, type, op, root, comm, stream));
    CUDACHECK(cudaSetDevice(savedDev));
    return ncclSuccess;
  } else {
    NCCLCHECK(ArgsCheck(sendbuff, recvbuff, count, type, op, root, comm, primName));
    NCCLCHECK(ncclCpuBarrierCheckin(comm));
    NCCLCHECK(ncclCpuBarrierWait(comm));
    return func(sendbuff, recvbuff, count, type, op, root, comm, stream);
  }
}
