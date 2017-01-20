/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "transport.h"
#include <cuda_runtime.h>
#include "net.h"
#include "gdcopy.h"
#include <assert.h>

struct netInfo {
  int rank;
};

struct netConnectInfo {
  ncclNetHandle_t netHandle;
};

struct netSendResources {
  void* netSendComm;
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
  struct ncclSendRecvMem* hostDevMem;
};

#define MAXSTEPS 8

struct netRecvResources {
  void* netRecvComm;
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
  struct ncclSendRecvMem* hostDevMem;
};

/* Fill information necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t netFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  struct netInfo* info = (struct netInfo*)opaqueInfo;
  static_assert(sizeof(struct netInfo) <= sizeof(ncclTinfo_t), "NET Info too large");
  info->rank = rank;
  return ncclSuccess;
}

/* Determine if we can communicate with the peer */
ncclResult_t netCanConnect(int* ret, ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo) {
  *ret = 1;
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */

void connectScattered(int nranks, int* groups, int group, int nextGroup, int* src, int* dst, int steps) {
  *src = groupPos(nranks, groups, group, steps+1);
  *dst = groupPos(nranks, groups, nextGroup, steps);
}

ncclResult_t netGetRings(int nranks, int ngroups, int* groups, int* values, int* nringsRet, int* prev, int* next, int pattern) {
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
ncclResult_t netSendSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct netSendResources* resources = (struct netSendResources*) malloc(sizeof(struct netSendResources));
  ring->send.transportResources = resources;
  resources->hostDevMem = NULL; //(struct ncclSendRecvMem*)gdptr(ring->devMem, ring->buffSize);

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  CUDACHECK(cudaHostAlloc(&resources->hostMem, size, cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));

  return ncclSuccess;
}

ncclResult_t netRecvSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct netRecvResources* resources = (struct netRecvResources*) malloc(sizeof(struct netRecvResources));
  ring->recv.transportResources = resources;
  resources->hostDevMem = NULL; //(struct ncclSendRecvMem*)gdptr(ring->devMem, ring->buffSize);

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  CUDACHECK(cudaHostAlloc(&resources->hostMem, size, cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));
  
  struct netInfo* myInfo = (struct netInfo*)myOpaqueInfo;
  struct netInfo* peerInfo = (struct netInfo*)peerOpaqueInfo;
  INFO("%d -> %d via NET/%s%s", peerInfo->rank, myInfo->rank, ncclNetName(), (resources->hostDevMem != NULL) ? "/GDCopy" : "");
  struct netConnectInfo* info = (struct netConnectInfo*) connectInfo;
  NCCLCHECK(ncclNetGetHandle(&info->netHandle, &resources->netRecvComm));
  return ncclSuccess;
}

ncclResult_t netSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct netSendResources* resources = (struct netSendResources*)send->transportResources;
  send->conn.buff = resources->devHostMem->buff;
  send->conn.tail = &resources->devHostMem->tail;
  send->conn.opCount = &resources->devHostMem->opCount;
  send->conn.fifo = resources->devHostMem->sizesFifo;

  if (resources->hostDevMem == NULL)
    send->conn.head = &resources->devHostMem->head;

  // Setup remote MPI rank / tag
  struct netConnectInfo* info = (struct netConnectInfo*)connectInfo;
  NCCLCHECK(ncclNetConnectHandle(info->netHandle, &resources->netSendComm));
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t netRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct netRecvResources* resources = (struct netRecvResources*)recv->transportResources;
  recv->conn.head = &resources->devHostMem->head;

  if (resources->hostDevMem == NULL) {
    recv->conn.buff = resources->devHostMem->buff;
    recv->conn.tail = &resources->devHostMem->tail;
    recv->conn.opCount = &resources->devHostMem->opCount;
  }

  // Setup remote MPI rank / tag
  return ncclSuccess;
}

ncclResult_t netSendFree(void* transportResources) {
  struct netSendResources* resources = (struct netSendResources*)transportResources;
  CUDACHECK(cudaFreeHost(resources->hostMem));
  // TODO : unmap hostDevMem
  free(resources);
  return ncclSuccess;
}

ncclResult_t netRecvFree(void* transportResources) {
  struct netRecvResources* resources = (struct netRecvResources*)transportResources;
  CUDACHECK(cudaFreeHost(resources->hostMem));
  // TODO : unmap hostDevMem
  free(resources);
  return ncclSuccess;
}

ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct netSendResources* resources = (struct netSendResources*) (ring->send.transportResources);
  volatile int* prevTail = &resources->hostMem->tail;
  int* prevHead = resources->hostDevMem ? &resources->hostDevMem->head : &resources->hostMem->head;
  char* localBuff = resources->hostMem->buff;
  int* sizesFifo = resources->hostMem->sizesFifo;
  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  // Update in case we skipped some collectives
  resources->hostMem->opCount = args->opCount;

  int head = 0;
  int tail = 0;

  int idle = 0;
  void* requests[args->substeps];
  while (tail < args->nsteps) {
    idle++;
    while (head != *prevTail) {
      // Send through MPI
      int slot = head%args->substeps;
      NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+slot*sliceSize, sizesFifo[slot], requests+slot));
      head++;
      idle = 0;
    }
    if (tail < head) {
      int done;
      int slot = tail%args->substeps;
      NCCLCHECK(ncclNetTest(requests[slot], &done, NULL));
      if (done) {
        tail++;
        *prevHead = tail;
        idle = 0;
      }
      if (idle) transportProxyIdle(idle);
    }
  }

  // Reset
  *prevTail = 0;
  resources->hostMem->opCount = args->opCount+1;
  return ncclSuccess;
}

ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct netRecvResources* resources = (struct netRecvResources*) (ring->recv.transportResources);

  assert(MAXSTEPS >= args->substeps);

  int* nextOpCount = resources->hostDevMem ? &resources->hostDevMem->opCount : &resources->hostMem->opCount;
  transportProxyWait([=] { return *nextOpCount >= args->opCount; });

  volatile int* nextHead = &resources->hostMem->head;
  char* localBuff = resources->hostMem->buff;
  char* nextBuff = resources->hostDevMem ? resources->hostDevMem->buff : NULL;
  int* nextTail = resources->hostDevMem ? &resources->hostDevMem->tail : &resources->hostMem->tail;

  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  int head = 0;
  int tail = 0;

  int idle = 0;
  void* requests[args->substeps];
  while (*nextHead < args->nsteps) {
    idle++;
    if ((*nextHead > tail - args->substeps) && (tail < args->nsteps)) {
      int slot = tail%args->substeps;
      NCCLCHECK(ncclNetIrecv(resources->netRecvComm, localBuff+slot*sliceSize, sliceSize, requests+slot));
      tail++;
      idle = 0;
    }
    if (tail > head) {
      int done;
      int slot = head%args->substeps;
      int size;
      NCCLCHECK(ncclNetTest(requests[slot], &done, &size));
      if (done) {
        if (nextBuff) memcpy(nextBuff+slot*sliceSize, localBuff+slot*sliceSize, size);
        head++;
        *nextTail = head;
      }
      idle = 0;
    }
    if (idle) transportProxyIdle(idle);
  }

  // Wait for last ack and reset
  transportProxyWait([=] { return *nextHead == head; });
  *nextHead = 0;

  return ncclSuccess;
}

struct ncclTransport netTransport = {
  "NET",
  netFillInfo,
  netCanConnect,
  netGetRings,
  { netSendSetup, netSendConnect, netSendFree, netSendProxy },
  { netRecvSetup, netRecvConnect, netRecvFree, netRecvProxy }
};
