/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "transport.h"
#include "nvmlwrap.h"
#include "net.h"
#include "gdcopy.h"
#include <cuda_runtime.h>
#include <assert.h>

#define NET_MAX_IFS 8
struct netInfo {
  int rank;
  int ndev;
  int scores[NET_MAX_IFS];
};

struct netConnectInfo {
  ncclNetHandle_t netHandle;
};

struct netSendResources {
  void* netSendComm;
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
  struct ncclSendRecvMem* hostDevMem;
  int netDev;
  bool cudaSupport;
  struct ncclSendRecvMem* devNetMem;
};

struct netRecvResources {
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
  struct ncclSendRecvMem* hostDevMem;
  int netDev;
  bool cudaSupport;
};

/* Fill information necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t netFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  struct netInfo* info = (struct netInfo*)opaqueInfo;
  static_assert(sizeof(struct netInfo) <= sizeof(ncclTinfo_t), "NET Info too large");
  info->rank = rank;
  int *distances;
  NCCLCHECK(ncclNetDevices(&info->ndev, &distances));
  if (info->ndev == 0) {
    WARN("Error : Network returned 0 device");
    return ncclSystemError;
  }
  if (info->ndev > NET_MAX_IFS) info->ndev = NET_MAX_IFS;
  for (int d=0; d<info->ndev; d++) info->scores[d] = distances[d];
  free(distances);
  return ncclSuccess;
}

/* Determine if we can communicate with the peer */
ncclResult_t netCanConnect(int* ret, ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo) {
  ret[0] = 0;
  struct netInfo* myInfo = (struct netInfo*)myOpaqueInfo;
  for (int d=0; d<myInfo->ndev; d++) {
    // Keep 2 bits of distance
    ret[0] |= ((myInfo->scores[d]&0x7)<<(3*d));
  }
  return ncclSuccess;
}

static inline void groupLastFirst(int nranks, int* groups, int group1, int group2, int* values, int ring, int* rank1, int* rank2, int minScore) {
  // Find last of group1
  for (int r1 = nranks-1; r1>=0; r1--) {
    if (groups[r1] == group1) {
      // Find first of group2
      for (int r2 = 0; r2<nranks; r2++) {
        if (groups[r2] == group2) {
          // Check both can talk through that device = ring
          if ((values[r1*nranks+r2] & (0x7<<(3*ring))) >= minScore &&
              (values[r2*nranks+r1] & (0x7<<(3*ring))) >= minScore) {
            *rank1 = r1;
            *rank2 = r2;
            return;
          }
        }
      }
    }
  }
  *rank1 = -1;
  *rank2 = -1;
}

ncclResult_t netGetRings(int nranks, int ngroups, int* groups, int* values, int* nringsRet, int* prev, int* next, int minScore) {
  for (int ring = 0; ring<*nringsRet; ring++) {
    for (int group = 0; group<ngroups; group++) {
      // Check if this group is already connected
      int skip = 0;
      for (int rank = 0; rank<nranks; rank++) {
        if (groups[rank] == group && next[ring*nranks+rank] != -1) skip = 1;
      }
      if (skip) continue;

      int source = -1, destination = -1;
      if (ring % 2 == 0) {
        int nextGroup = (group+1)%ngroups;
        groupLastFirst(nranks, groups, group, nextGroup, values, ring, &source, &destination, minScore);
      } else {
        int prevGroup = (group-1+ngroups)%ngroups;
        groupLastFirst(nranks, groups, prevGroup, group, values, ring, &destination, &source, minScore);
      }
      if (source == -1 || destination == -1) {
        *nringsRet = ring;
        return ncclSuccess;
      }
      next[ring*nranks+source] = destination;
      prev[ring*nranks+destination] = source;
    }
  }
  return ncclSuccess;
}

static ncclResult_t netHostAlloc(struct ncclSendRecvMem** ptr, size_t size) {
  // Allocate memory close to the device we are using
  CUDACHECK(cudaHostAlloc(ptr, size, cudaHostAllocMapped));
  return ncclSuccess;
}

int getDev(int ringId, int nDev, int* scores) {
  int maxScore = 0;
  for (int d=0; d<nDev; d++) if (scores[d] > maxScore) maxScore = scores[d];
  int skip = ringId+1;
  while (skip) {
    for (int d=0; d<nDev; d++) {
      if (scores[d] == maxScore) {
        skip--;
        if (skip == 0) return d;
      }
    }
  }
  return 0;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t netSendSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct netSendResources* resources = (struct netSendResources*) malloc(sizeof(struct netSendResources));
  ring->send.transportResources = resources;
  resources->hostDevMem = NULL; //(struct ncclSendRecvMem*)gdptr(ring->devMem, ring->buffSize);

  struct netInfo* myInfo = (struct netInfo*)myOpaqueInfo;
  resources->netDev = getDev(ring->id, myInfo->ndev, myInfo->scores);
  int flags;
  NCCLCHECK(ncclNetPtrSupport(resources->netDev, &flags));
  static int useGDRforReads = -1;
  if (useGDRforReads == -1) {
    char* str = getenv("NCCL_NET_GDR_READ");
    useGDRforReads = str ? atoi(str) : 0;
  }
  resources->cudaSupport = (useGDRforReads == 1) && (flags & NCCL_PTR_CUDA) ? true : false;

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  if (resources->cudaSupport) {
    CUDACHECK(cudaMalloc(&resources->devNetMem, size));
  }
  NCCLCHECK(netHostAlloc(&resources->hostMem, size));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));

  return ncclSuccess;
}

ncclResult_t netRecvSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct netRecvResources* resources = (struct netRecvResources*) malloc(sizeof(struct netRecvResources));
  ring->recv.transportResources = resources;
  resources->hostDevMem = NULL; //(struct ncclSendRecvMem*)gdptr(ring->devMem, ring->buffSize);

  struct netInfo* myInfo = (struct netInfo*)myOpaqueInfo;
  resources->netDev = getDev(ring->id, myInfo->ndev, myInfo->scores);
  int flags;
  NCCLCHECK(ncclNetPtrSupport(resources->netDev, &flags));
  resources->cudaSupport = (flags & NCCL_PTR_CUDA) ? true : false;

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  NCCLCHECK(netHostAlloc(&resources->hostMem, size));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));
  
  struct netInfo* peerInfo = (struct netInfo*)peerOpaqueInfo;
  INFO("%d -> %d via NET/%s/%d%s%s", peerInfo->rank, myInfo->rank, ncclNetName(), resources->netDev,
      resources->cudaSupport ? "/GDRDMA" : "", 
      (resources->hostDevMem != NULL) ? "/GDCopy" : "");
  struct netConnectInfo* info = (struct netConnectInfo*) connectInfo;
  NCCLCHECK(ncclNetListen(resources->netDev, &info->netHandle, &resources->netListenComm));
  return ncclSuccess;
}

ncclResult_t netSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct netSendResources* resources = (struct netSendResources*)send->transportResources;

  if (resources->cudaSupport) {
    send->conn.buff = resources->devNetMem->buff;
  } else {
    send->conn.buff = resources->devHostMem->buff;
  }
  send->conn.tail = &resources->devHostMem->tail;
  send->conn.opCount = &resources->devHostMem->opCount;
  send->conn.fifo = resources->devHostMem->sizesFifo;

  if (resources->hostDevMem == NULL)
    send->conn.head = &resources->devHostMem->head;

  // Connect to remote peer
  struct netConnectInfo* info = (struct netConnectInfo*)connectInfo;
  NCCLCHECK(ncclNetConnect(resources->netDev, info->netHandle, &resources->netSendComm));
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t netRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct netRecvResources* resources = (struct netRecvResources*)recv->transportResources;

  recv->conn.head = &resources->devHostMem->head;

  if (resources->cudaSupport == false)
    recv->conn.buff = resources->devHostMem->buff;

  if (resources->hostDevMem == NULL) {
    recv->conn.tail = &resources->devHostMem->tail;
    recv->conn.opCount = &resources->devHostMem->opCount;
  }

  // Finish connection establishment
  NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
  NCCLCHECK(ncclNetCloseListen(resources->netListenComm));

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
  char* localBuff = resources->cudaSupport ? resources->devNetMem->buff : resources->hostMem->buff;
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
      // Send through network
      int slot = head%args->substeps;
      NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+slot*sliceSize, sizesFifo[slot], NCCL_PTR_HOST, requests+slot));
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

  int* nextOpCount = resources->hostDevMem ? &resources->hostDevMem->opCount : &resources->hostMem->opCount;
  transportProxyWait([=] { return *nextOpCount >= args->opCount; });

  volatile int* nextHead = &resources->hostMem->head;
  int ptrType = resources->cudaSupport ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
  char* localBuff = resources->cudaSupport ? ring->devMem->buff : resources->hostMem->buff;
  char* nextBuff = (resources->cudaSupport == false && resources->hostDevMem) ? resources->hostDevMem->buff : NULL;
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
      NCCLCHECK(ncclNetIrecv(resources->netRecvComm, localBuff+slot*sliceSize, sliceSize, ptrType, requests+slot));
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
