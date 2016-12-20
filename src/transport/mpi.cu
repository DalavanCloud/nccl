/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "transport.h"
#include <cuda_runtime.h>
#include "mpi.h"
#include "gdcopy.h"
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
  struct ncclSendRecvMem* hostDevMem;
};

#define MAXSTEPS 8

struct mpiResourcesRecv {
  int mpiRank;
  int mpiTag;
  cudaStream_t stream;
  cudaEvent_t syncEvent[MAXSTEPS];
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
  struct ncclSendRecvMem* hostDevMem;
};

/* Fill information necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t mpiFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  struct mpiInfo* info = (struct mpiInfo*)opaqueInfo;
  static_assert(sizeof(struct mpiInfo) <= sizeof(ncclTinfo_t), "MPI Info too large");
  info->rank = rank;
  return ncclSuccess;
}

/* Determine if we can communicate with the peer */
ncclResult_t mpiCanConnect(int* ret, ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo) {
  *ret = ncclMpiEnabled();
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */

void connectScattered(int nranks, int* groups, int group, int nextGroup, int* src, int* dst, int steps) {
  *src = groupPos(nranks, groups, group, steps+1);
  *dst = groupPos(nranks, groups, nextGroup, steps);
}

ncclResult_t mpiGetRings(int nranks, int ngroups, int* groups, int* values, int* nringsRet, int* prev, int* next, int pattern) {
  if (pattern >= 2) {
    *nringsRet = 0;
    return ncclSuccess;
  }
  *nringsRet = 1;
  for (int ring = 0; ring<*nringsRet; ring++) {
    for (int group = 0; group<ngroups; group++) {
      // Check if this group is already connected
      int skip = 0;
      for (int rank = 0; rank<nranks; rank++) {
        if (groups[rank] == group && next[ring*nranks+rank] != -1) skip = 1;
      }
      if (skip) continue;

      int nextGroup = (group+1)%ngroups;
      int source = -1, destination = -1;
      if (pattern == 0) {
        if (ring % 2 == 0) {
          source = groupLast(nranks, groups, group);
          destination = groupFirst(nranks, groups, nextGroup);
        } else {
          source = groupFirst(nranks, groups, group);
          destination = groupLast(nranks, groups, nextGroup);
        }
      } else if (pattern == 1) {
        source = groupPos(nranks, groups, group, ring*2+1);
        destination = groupPos(nranks, groups, nextGroup, ring*2);
      }
      if (source == -1 || destination == -1) {
        WARN("source %d dest %d, stopping\n", source, destination);
        *nringsRet = ring;
        return ncclSuccess;
      }
      next[ring*nranks+source] = destination;
      prev[ring*nranks+destination] = source;
    }
  }
  return ncclSuccess;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t mpiSetupSend(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct mpiResourcesSend* resources = (struct mpiResourcesSend*) malloc(sizeof(struct mpiResourcesSend));
  ring->send.transportResources = resources;
  resources->hostDevMem = (struct ncclSendRecvMem*)gdptr(ring->devMem, ring->buffSize);

  struct mpiInfo* info = (struct mpiInfo*)myOpaqueInfo;
  MPICHECK(ncclMpiCommRank(&info->mpiRank));

  // Create stream for proxy
  CUDACHECK(cudaStreamCreateWithFlags(&resources->stream, cudaStreamNonBlocking));

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  CUDACHECK(cudaHostAlloc(&resources->hostMem, size, cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));

  memcpy(connectInfo, info, sizeof(struct mpiInfo));
  return ncclSuccess;
}

ncclResult_t mpiSetupRecv(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct mpiResourcesRecv* resources = (struct mpiResourcesRecv*) malloc(sizeof(struct mpiResourcesRecv));
  ring->recv.transportResources = resources;
  resources->hostDevMem = (struct ncclSendRecvMem*)gdptr(ring->devMem, ring->buffSize);

  struct mpiInfo* info = (struct mpiInfo*)myOpaqueInfo;
  MPICHECK(ncclMpiCommRank(&info->mpiRank));
  // Allocate a tag for this peer
  MPICHECK(ncclMpiGetTag(&info->mpiTag));
  resources->mpiTag = info->mpiTag;

  // Create stream for proxy
  CUDACHECK(cudaStreamCreateWithFlags(&resources->stream, cudaStreamNonBlocking));
  // And event
  for (int i=0; i<MAXSTEPS; i++)
    CUDACHECK(cudaEventCreate(resources->syncEvent+i));

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  CUDACHECK(cudaHostAlloc(&resources->hostMem, size, cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));
  
  struct mpiInfo* peerInfo = (struct mpiInfo*)peerOpaqueInfo;
  INFO("%d -> %d via MPI%s%s", peerInfo->rank, info->rank, ncclMpiCudaSupport() ? "/GDRDMA" : "", (resources->hostDevMem != NULL) ? "/GDCopy" : "");
  memcpy(connectInfo, info, sizeof(struct mpiInfo));
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

// Don't use the same requests for Send and Recv 
#define SEND_REQ(slot) (2*slot)
#define RECV_REQ(slot) (2*slot+1)

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
  int data[args->substeps];

  // Update in case we skipped some collectives
  resources->hostMem->opCount = args->opCount;

  int tail = 0;
  int idle = 0;
  while (tail < args->nsteps) {
    idle++;
    while (head != *prevTail) {
      // Send through MPI
      int slot = head%args->substeps;
      MPICHECK(ncclMpiIsend(resources->mpiRank, localBuff+slot*sliceSize, maxSize, resources->mpiTag, SEND_REQ(slot)));
      head++;
      idle = 0;
    }
    if (tail < head) {
      int done;
      int slot = tail%args->substeps;
      MPICHECK(ncclMpiTest(SEND_REQ(slot), &done));
      if (done) {
        tail++;
        data[slot] = tail;
	CUDACHECK(cudaMemcpyAsync(prevHead, data+slot, sizeof(int), cudaMemcpyHostToDevice, resources->stream));
        idle = 0;
      }
      if (idle) transportProxyIdle(idle);
    }
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

  int mpiCudaSupport = ncclMpiCudaSupport();
  bool directDevMem = resources->hostDevMem != NULL;

  assert(MAXSTEPS >= args->substeps);

  if (directDevMem) {
    int* nextOpCount = &resources->hostDevMem->opCount;
    transportProxyWait([=] { return *nextOpCount >= args->opCount; });
  } else {
    int val = 0;
    int* nextOpCount = &devMem->opCount;
    while (val != args->opCount) {
      CUDACHECK(cudaMemcpyAsync(&val, nextOpCount, sizeof(int), cudaMemcpyDeviceToHost, resources->stream));
      CUDACHECK(cudaStreamSynchronize(resources->stream));
    }
  }

  volatile int* nextHead = &resources->hostMem->head;
  int* nextTail = (mpiCudaSupport && directDevMem) ? &resources->hostDevMem->tail : &devMem->tail;
  char* localBuff = resources->hostMem->buff;
  char* nextBuff = devMem->buff;

  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;
  int maxSize = min(sliceSize, args->size);

  int head = 0;
  int data[args->substeps];

  int tail = 0;
  int idle = 0;
  while (tail < args->nsteps) {
    idle++;
    if (mpiCudaSupport == 1) {
      while (((head - *nextHead) < args->substeps) && (head < args->nsteps)) {
        int slot = head%args->substeps;
        MPICHECK(ncclMpiIrecv(resources->mpiRank, nextBuff+slot*sliceSize, maxSize, resources->mpiTag, RECV_REQ(slot)));
        head++;
        idle = 0;
      }
      if (tail < head) {
        int done;
        int slot = tail%args->substeps;
        MPICHECK(ncclMpiTest(RECV_REQ(slot), &done));
        if (done) {
          tail++;
          if (directDevMem) {
            *nextTail = tail;
          } else {
            data[slot] = tail;
            CUDACHECK(cudaMemcpyAsync(nextTail, data+slot, sizeof(int), cudaMemcpyHostToDevice, resources->stream));
          }
          idle = 0;
        }
      }
    } else {
      if (((head - tail) < args->substeps) && (head < args->nsteps)) {
        int slot = head%args->substeps;
        if (cudaEventQuery(resources->syncEvent[slot]) == cudaSuccess) {
          MPICHECK(ncclMpiIrecv(resources->mpiRank, localBuff+slot*sliceSize, maxSize, resources->mpiTag, RECV_REQ(slot)));
          head++;
          idle = 0;
        }
      }
      if (tail < head && ((tail - *nextHead) < args->substeps)) {
        int done;
        int slot = tail%args->substeps;
        MPICHECK(ncclMpiTest(RECV_REQ(slot), &done));
        if (done) {
          // Send to GPU
          CUDACHECK(cudaMemcpyAsync(nextBuff+slot*sliceSize, localBuff+slot*sliceSize, maxSize, cudaMemcpyHostToDevice, resources->stream));
          CUDACHECK(cudaEventRecord(resources->syncEvent[slot], resources->stream));
          tail++;
          data[slot] = tail;
          CUDACHECK(cudaMemcpyAsync(nextTail, data+slot, sizeof(int), cudaMemcpyHostToDevice, resources->stream));
        }
        idle = 0;
      }
    }
    if (idle) transportProxyIdle(idle);
  }
  // Ensure all updates are pushed
  CUDACHECK(cudaStreamSynchronize(resources->stream));

  // Wait for last ack and reset
  transportProxyWait([=] { return *nextHead == head; });
  *nextHead = 0;

  return ncclSuccess;
}

struct ncclTransport mpiTransport = {
  "MPI",
  mpiFillInfo,
  mpiCanConnect,
  mpiGetRings,
  { mpiSetupSend, mpiConnectSend, mpiSendProxy },
  { mpiSetupRecv, mpiConnectRecv, mpiRecvProxy }
};
