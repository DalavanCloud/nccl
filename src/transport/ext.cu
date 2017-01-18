/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "transport.h"
#include <cuda_runtime.h>
#include "ext.h"
#include "gdcopy.h"
#include <assert.h>

struct extInfo {
  int rank;
};

struct extConnectInfo {
  ncclExtHandle_t extHandle;
};

struct extSendResources {
  void* extSendComm;
  cudaStream_t stream;
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
  struct ncclSendRecvMem* hostDevMem;
};

#define MAXSTEPS 8

struct extRecvResources {
  void* extRecvComm;
  cudaStream_t stream;
  cudaEvent_t syncEvent[MAXSTEPS];
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
  struct ncclSendRecvMem* hostDevMem;
};

/* Fill information necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t extFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  struct extInfo* info = (struct extInfo*)opaqueInfo;
  static_assert(sizeof(struct extInfo) <= sizeof(ncclTinfo_t), "EXT Info too large");
  info->rank = rank;
  return ncclSuccess;
}

/* Determine if we can communicate with the peer */
ncclResult_t extCanConnect(int* ret, ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo) {
  *ret = ncclExtEnabled() == ncclSuccess ? 1 : 0;
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */

void connectScattered(int nranks, int* groups, int group, int nextGroup, int* src, int* dst, int steps) {
  *src = groupPos(nranks, groups, group, steps+1);
  *dst = groupPos(nranks, groups, nextGroup, steps);
}

ncclResult_t extGetRings(int nranks, int ngroups, int* groups, int* values, int* nringsRet, int* prev, int* next, int pattern) {
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
ncclResult_t extSendSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct extSendResources* resources = (struct extSendResources*) malloc(sizeof(struct extSendResources));
  ring->send.transportResources = resources;
  resources->hostDevMem = (struct ncclSendRecvMem*)gdptr(ring->devMem, ring->buffSize);

  // Create stream for proxy
  CUDACHECK(cudaStreamCreateWithFlags(&resources->stream, cudaStreamNonBlocking));

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  CUDACHECK(cudaHostAlloc(&resources->hostMem, size, cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));

  return ncclSuccess;
}

ncclResult_t extRecvSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct extRecvResources* resources = (struct extRecvResources*) malloc(sizeof(struct extRecvResources));
  ring->recv.transportResources = resources;
  resources->hostDevMem = (struct ncclSendRecvMem*)gdptr(ring->devMem, ring->buffSize);

  // Create stream for proxy
  CUDACHECK(cudaStreamCreateWithFlags(&resources->stream, cudaStreamNonBlocking));
  // And event
  for (int i=0; i<MAXSTEPS; i++)
    CUDACHECK(cudaEventCreate(resources->syncEvent+i));

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  CUDACHECK(cudaHostAlloc(&resources->hostMem, size, cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));
  
  struct extInfo* myInfo = (struct extInfo*)myOpaqueInfo;
  struct extInfo* peerInfo = (struct extInfo*)peerOpaqueInfo;
  INFO("%d -> %d via EXT%s%s", peerInfo->rank, myInfo->rank, ncclExtCudaSupport() ? "/GDRDMA" : "", (resources->hostDevMem != NULL) ? "/GDCopy" : "");
  struct extConnectInfo* info = (struct extConnectInfo*) connectInfo;
  NCCLCHECK(ncclExtGetHandle(&info->extHandle, &resources->extRecvComm));
  return ncclSuccess;
}

ncclResult_t extSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct extSendResources* resources = (struct extSendResources*)send->transportResources;
  send->conn.buff = resources->devHostMem->buff;
  send->conn.tail = &resources->devHostMem->tail;
  send->conn.opCount = &resources->devHostMem->opCount;
  send->conn.fifo = resources->devHostMem->sizesFifo;

  // Setup remote MPI rank / tag
  struct extConnectInfo* info = (struct extConnectInfo*)connectInfo;
  NCCLCHECK(ncclExtConnectHandle(info->extHandle, &resources->extSendComm));
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t extRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct extRecvResources* resources = (struct extRecvResources*)recv->transportResources;
  recv->conn.head = &resources->devHostMem->head;

  // Setup remote MPI rank / tag
  return ncclSuccess;
}

ncclResult_t extSendFree(void* transportResources) {
  struct extSendResources* resources = (struct extSendResources*)transportResources;
  CUDACHECK(cudaStreamDestroy(resources->stream));
  CUDACHECK(cudaFreeHost(resources->hostMem));
  // TODO : unmap hostDevMem
  free(resources);
  return ncclSuccess;
}

ncclResult_t extRecvFree(void* transportResources) {
  struct extRecvResources* resources = (struct extRecvResources*)transportResources;
  CUDACHECK(cudaStreamDestroy(resources->stream));
  for (int i=0; i<MAXSTEPS; i++) {
    CUDACHECK(cudaEventDestroy(resources->syncEvent[i]));
  }
  CUDACHECK(cudaFreeHost(resources->hostMem));
  // TODO : unmap hostDevMem
  free(resources);
  return ncclSuccess;
}

ncclResult_t extSendProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct extSendResources* resources = (struct extSendResources*) (ring->send.transportResources);
  struct ncclSendRecvMem* devMem = ring->devMem;
  volatile int* prevTail = &resources->hostMem->tail;
  int* prevHead = &devMem->head;
  char* localBuff = resources->hostMem->buff;
  int* sizesFifo = resources->hostMem->sizesFifo;
  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  int head = 0;
  int data[args->substeps];

  // Update in case we skipped some collectives
  resources->hostMem->opCount = args->opCount;

  int tail = 0;
  int idle = 0;
  void* requests[args->substeps];
  while (tail < args->nsteps) {
    idle++;
    while (head != *prevTail) {
      // Send through MPI
      int slot = head%args->substeps;
      NCCLCHECK(ncclExtIsend(resources->extSendComm, localBuff+slot*sliceSize, sizesFifo[slot], requests+slot));
      head++;
      idle = 0;
    }
    if (tail < head) {
      int done;
      int slot = tail%args->substeps;
      NCCLCHECK(ncclExtTest(requests[slot], &done, NULL));
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

ncclResult_t extRecvProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct extRecvResources* resources = (struct extRecvResources*) (ring->recv.transportResources);
  struct ncclSendRecvMem* devMem = ring->devMem;

  int extCudaSupport = ncclExtCudaSupport();
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
  int* nextTail = (extCudaSupport && directDevMem) ? &resources->hostDevMem->tail : &devMem->tail;
  char* localBuff = resources->hostMem->buff;
  char* nextBuff = devMem->buff;

  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  int head = 0;
  int data[args->substeps];

  int tail = 0;
  int idle = 0;
  void* requests[args->substeps];
  while (tail < args->nsteps) {
    idle++;
    if (extCudaSupport == 1) {
      while (((head - *nextHead) < args->substeps) && (head < args->nsteps)) {
        int slot = head%args->substeps;
        NCCLCHECK(ncclExtIrecv(resources->extRecvComm, nextBuff+slot*sliceSize, sliceSize, requests+slot));
        head++;
        idle = 0;
      }
      if (tail < head) {
        int done;
        int slot = tail%args->substeps;
        NCCLCHECK(ncclExtTest(requests[slot], &done, NULL));
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
          NCCLCHECK(ncclExtIrecv(resources->extRecvComm, localBuff+slot*sliceSize, sliceSize, requests+slot));
          head++;
          idle = 0;
        }
      }
      if (tail < head && ((tail - *nextHead) < args->substeps)) {
        int done;
        int slot = tail%args->substeps;
        int size;
        NCCLCHECK(ncclExtTest(requests[slot], &done, &size));
        if (done) {
          // Send to GPU
          CUDACHECK(cudaMemcpyAsync(nextBuff+slot*sliceSize, localBuff+slot*sliceSize, size, cudaMemcpyHostToDevice, resources->stream));
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

struct ncclTransport extTransport = {
  "EXT",
  extFillInfo,
  extCanConnect,
  extGetRings,
  { extSendSetup, extSendConnect, extSendFree, extSendProxy },
  { extRecvSetup, extRecvConnect, extRecvFree, extRecvProxy }
};
