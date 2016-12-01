/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "transport.h"
#include <cuda_runtime.h>
#include "mpi.h"
#include <assert.h>

struct mpiInfo {
  int rank;
  int mpiRank;
  int mpiTag;
};

struct mpiResourcesSend {
  int mpiRank;
  int mpiTag;
  cudaStream_t stream;
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
};

#define MAXSTEPS 8

struct mpiResourcesRecv {
  int mpiRank;
  int mpiTag;
  cudaStream_t stream;
  cudaEvent_t syncEvent[MAXSTEPS];
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
};

/* Fill infomation necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t mpiFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  return ncclSuccess;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t mpiSetupSend(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring, int* select) {
  struct mpiResourcesSend* resources = (struct mpiResourcesSend*) malloc(sizeof(struct mpiResourcesSend));
  ring->send.transportResources = resources;

  struct mpiInfo info;
  MPICHECK(ncclMpiCommRank(&info.mpiRank));

  // Create stream for proxy
  CUDACHECK(cudaStreamCreateWithFlags(&resources->stream, cudaStreamNonBlocking));

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  CUDACHECK(cudaHostAlloc(&resources->hostMem, size, cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));

  memcpy(connectInfo, &info, sizeof(struct mpiInfo));
  *select = 1;
  return ncclSuccess;
}

ncclResult_t mpiSetupRecv(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring, int* select) {
  struct mpiResourcesRecv* resources = (struct mpiResourcesRecv*) malloc(sizeof(struct mpiResourcesRecv));
  ring->recv.transportResources = resources;

  struct mpiInfo info;
  MPICHECK(ncclMpiCommRank(&info.mpiRank));
  // Allocate a tag for this peer
  MPICHECK(ncclMpiGetTag(&info.mpiTag));
  resources->mpiTag = info.mpiTag;

  // Create stream for proxy
  CUDACHECK(cudaStreamCreateWithFlags(&resources->stream, cudaStreamNonBlocking));
  // And event
  for (int i=0; i<MAXSTEPS; i++)
    CUDACHECK(cudaEventCreate(resources->syncEvent+i));

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  CUDACHECK(cudaHostAlloc(&resources->hostMem, size, cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));
  
  memcpy(connectInfo, &info, sizeof(struct mpiInfo));
  *select = 1;
  return ncclSuccess;
}

ncclResult_t mpiConnectSend(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct mpiResourcesSend* resources = (struct mpiResourcesSend*)send->transportResources;
  send->conn.buff = resources->devHostMem->buff;
  send->conn.tail = &resources->devHostMem->tail;
  send->conn.opCount = &resources->devHostMem->opCount;

  // Setup remote MPI rank / tag
  struct mpiInfo* info = (struct mpiInfo*)connectInfo;
  resources->mpiRank = info->mpiRank;
  resources->mpiTag = info->mpiTag;
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t mpiConnectRecv(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct mpiResourcesRecv* resources = (struct mpiResourcesRecv*)recv->transportResources;
  recv->conn.head = &resources->devHostMem->head;

  // Setup remote MPI rank / tag
  struct mpiInfo* info = (struct mpiInfo*)connectInfo;
  resources->mpiRank = info->mpiRank;
  return ncclSuccess;
}

ncclResult_t mpiSendProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct mpiResourcesSend* resources = (struct mpiResourcesSend*) (ring->send.transportResources);
  struct ncclSendRecvMem* devMem = ring->devMem;
  volatile int* prevTail = &resources->hostMem->tail;
  int* prevHead = &devMem->head;
  char* localBuff = resources->hostMem->buff;
  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;
  int maxSize = min(sliceSize, args->size);

  int head = 0;
  int offset = 0;

  // Update in case we skipped some collectives
  resources->hostMem->opCount = args->opCount;

  //printf("%d steps of %d size\n", args->nsteps, maxSize);
  while (head < args->nsteps) {
    // Receive from GPU
    transportProxyWait([=] { return head != *prevTail; });

    // Send to mpi
    MPICHECK(ncclMpiSend(resources->mpiRank, localBuff+offset, maxSize, resources->mpiTag));
    head++;
    CUDACHECK(cudaMemcpyAsync(prevHead, &head, sizeof(int), cudaMemcpyHostToDevice, resources->stream));

    offset += sliceSize;
    if (offset == buffSize)
      offset = 0;
  }
  // Ensure all updates are pushed
  CUDACHECK(cudaStreamSynchronize(resources->stream));

  // Reset
  *prevTail = 0;
  resources->hostMem->opCount = args->opCount+1;
  return ncclSuccess;
}

ncclResult_t mpiRecvProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct mpiResourcesRecv* resources = (struct mpiResourcesRecv*) (ring->recv.transportResources);
  struct ncclSendRecvMem* devMem = ring->devMem;
  int* nextTail = &devMem->tail;
  int* nextOpCount = &devMem->opCount;
  volatile int* nextHead = &resources->hostMem->head;
  char* localBuff = resources->hostMem->buff;
  char* nextBuff = devMem->buff;
  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;
  int maxSize = min(sliceSize, args->size);
  assert(MAXSTEPS >= args->substeps);

  int val = 0;
  while (val != args->opCount) {
    CUDACHECK(cudaMemcpyAsync(&val, nextOpCount, sizeof(int), cudaMemcpyDeviceToHost, resources->stream));
    CUDACHECK(cudaStreamSynchronize(resources->stream));
  }
  int head = 0;
  int offset = 0;
  int mpiCudaSupport = ncclMpiCudaSupport();

  while (head < args->nsteps) {
    // Receive from mpi
    if (mpiCudaSupport == 1) {
      transportProxyWait([=] { return (head - *nextHead) < args->substeps; });
      MPICHECK(ncclMpiRecv(resources->mpiRank, nextBuff+offset, maxSize, resources->mpiTag));
    } else {
      CUDACHECK(cudaEventSynchronize(resources->syncEvent[head%args->substeps]));
      MPICHECK(ncclMpiRecv(resources->mpiRank, localBuff+offset, maxSize, resources->mpiTag));
      // Send to GPU
      transportProxyWait([=] { return (head - *nextHead) < args->substeps; });
      CUDACHECK(cudaMemcpyAsync(nextBuff+offset, localBuff+offset, maxSize, cudaMemcpyHostToDevice, resources->stream));
      CUDACHECK(cudaEventRecord(resources->syncEvent[head%args->substeps], resources->stream));
    }
    head++;
    CUDACHECK(cudaMemcpyAsync(nextTail, &head, sizeof(int), cudaMemcpyHostToDevice, resources->stream));

    offset += sliceSize;
    if (offset == buffSize)
      offset = 0;
  }
  // Ensure all updates are pushed
  CUDACHECK(cudaStreamSynchronize(resources->stream));

  // Wait for last ack and reset
  transportProxyWait([=] { return *nextHead == head; });
  *nextHead = 0;

  return ncclSuccess;
}

struct ncclTransport mpiTransport = {
  mpiFillInfo,
  { mpiSetupSend, mpiConnectSend, mpiSendProxy },
  { mpiSetupRecv, mpiConnectRecv, mpiRecvProxy }
};
