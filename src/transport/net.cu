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
  uint64_t llStep;
  uint64_t llLastCleaning;
};

struct netRecvResources {
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
  struct ncclSendRecvMem* hostDevMem;
  int netDev;
  bool cudaSupport;
  uint64_t llStep;
  uint64_t llLastCleaning;
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

static inline int groupBestStart(int nranks, int* groups, int group, int* values, int card, int minScore) {
  int bestRank = -1;
  int bestScore = 0;
  for (int rank=0; rank<nranks; rank++) {
    if (groups[rank] != group) continue;
    for (int i=0; i<nranks; i++) {
      int netValue = values[rank*nranks+i];
      if (netValue != 0) {
        int score = (netValue>>(3*card)) & 0x7;
        if (score >= minScore && score > bestScore) {
          bestScore = score;
          bestRank = rank;
        }
        // All other values should be the same, stop here for this rank
        break;
      }
    }
  }
  return bestRank;
}
static inline int groupBestEnd(int nranks, int* groups, int group, int* subgroups, int startSubGroup, int startRank, int* values, int card, int minScore) {
  // For the last rank, we don't need the absolute best score, just to be within minScore.
  for (int rank=nranks-1; rank>=0; rank--) {
    if (groups[rank] != group) continue;
    if (startSubGroup != -1 && startSubGroup == subgroups[rank]) continue;
    if (startRank == rank) continue;
    for (int i=0; i<nranks; i++) {
      int netValue = values[rank*nranks+i];
      if (netValue != 0) {
        int score = (netValue>>(3*card)) & 0x7;
        if (score >= minScore) {
          return rank;
        }
        // All other values should be the same, stop here for this rank
        break;
      }
    }
  }
  return -1;
}


ncclResult_t netGetRings(int nranks, int* groups, int* subgroups, int* values, int* nringsRet, int* prev, int* next, int minScore, int* nthreads) {
  int nGroups = groups[nranks-1] + 1;
  int cardUsed[NET_MAX_IFS*nGroups];
  for (int c=0; c<NET_MAX_IFS*nGroups; c++) cardUsed[c] = 0;

  for (int ring = 0; ring<*nringsRet; ring++) {
    int starts[nGroups];
    int ends[nGroups];
    for (int group = 0; group<nGroups; group++) {
      int nranksInGroup = 0;
      int nsubGroups = 0;
      for (int rank=0; rank<nranks; rank++) if (groups[rank] == group) {
        nranksInGroup++;
        nsubGroups = max(subgroups[rank], nsubGroups);
      }
      starts[group] = ends[group] = -1;
      // Receive on the rank closest to the NIC
      for (int card=0; card<NET_MAX_IFS; card++) {
        if (cardUsed[group*NET_MAX_IFS+card] == 1) continue;
        int start = groupBestStart(nranks, groups, group, values, card, minScore);
        // Send from any rank, but best on a different subgroup and close to the NIC also.
        int end = (nranksInGroup == 1) ? start 
          : groupBestEnd(nranks, groups, group, subgroups, nsubGroups ? subgroups[start] : -1, start, values, card, minScore);
        //printf("Ring %d, Minscore %d, Card %d, group %d, start = %d, end = %d\n", ring, minScore, card, group, start, end);
        if (start != -1 && end != -1) {
          cardUsed[group*NET_MAX_IFS+card] = 1;
          starts[group] = start;
          ends[group] = end;
          break;
        }
      }
      if (starts[group] == -1 || ends[group] == -1) {
        *nringsRet = ring;
        return ncclSuccess;
      }
    }
    // Link groups together
    for (int group = 0; group<nGroups; group++) {
      int nextGroup = (group+1)%nGroups;
      next[ring*nranks+ends[group]] = starts[nextGroup];
      prev[ring*nranks+starts[nextGroup]] = ends[group];
    }
  }
  return ncclSuccess;
}

static ncclResult_t netHostAlloc(struct ncclSendRecvMem** ptr, size_t size) {
  // Allocate memory close to the device we are using
  CUDACHECK(cudaHostAlloc(ptr, size, cudaHostAllocMapped));
  memset(*ptr, 0, size);
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
  struct netSendResources* resources = (struct netSendResources*) mallocZero(sizeof(struct netSendResources));
  ring->send.transportResources = resources;
//  resources->hostDevMem = (struct ncclSendRecvMem*)gdptr(ring->devMem, ring->buffSize);

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
    CUDACHECK(cudaMemset(resources->devNetMem, 0, size));
  }
  NCCLCHECK(netHostAlloc(&resources->hostMem, size));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));

  return ncclSuccess;
}

ncclResult_t netRecvSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct netRecvResources* resources = (struct netRecvResources*) mallocZero(sizeof(struct netRecvResources));
  ring->recv.transportResources = resources;
//  resources->hostDevMem = (struct ncclSendRecvMem*)gdptr(ring->devMem, ring->buffSize);

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
    // We don't use devMem for llMode because the CPU has to read the data
    send->conn.llBuff = resources->devHostMem->llBuff;
  } else {
    send->conn.buff = resources->devHostMem->buff;
    send->conn.llBuff = resources->devHostMem->llBuff;
  }
  send->conn.tail = &resources->devHostMem->tail;
  send->conn.opCount = &resources->devHostMem->opCount;
  send->conn.fifo = resources->devHostMem->sizesFifo;
  send->conn.llFifo = resources->devHostMem->llSizesFifo;

  if (resources->hostDevMem == NULL) {
    send->conn.head = &resources->devHostMem->head;
    send->conn.llHead = &resources->devHostMem->llHead;
  }

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
  recv->conn.llHead = &resources->devHostMem->llHead;

  if (resources->cudaSupport == false) {
    recv->conn.buff = resources->devHostMem->buff;
    recv->conn.llBuff = resources->devHostMem->llBuff;
  }

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
  NCCLCHECK(ncclNetCloseSend(resources->netSendComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netRecvFree(void* transportResources) {
  struct netRecvResources* resources = (struct netRecvResources*)transportResources;
  CUDACHECK(cudaFreeHost(resources->hostMem));
  // TODO : unmap hostDevMem
  NCCLCHECK(ncclNetCloseRecv(resources->netRecvComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct netSendResources* resources = (struct netSendResources*) (ring->send.transportResources);
  const int llMode = args->llMode;

  volatile uint64_t* prevTail = &resources->hostMem->tail;
  struct ncclSendRecvMem* prevMem = resources->hostDevMem ? resources->hostDevMem : resources->hostMem;
  uint64_t* prevHead = llMode ? &prevMem->llHead : &prevMem->head;
  struct ncclSendRecvMem* localMem = resources->cudaSupport ? resources->devNetMem : resources->hostMem;
  char* localBuff = llMode ? resources->hostMem->llBuff : localMem->buff;
  int ptrType = resources->cudaSupport ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
  volatile int* sizesFifo = llMode ? resources->hostMem->llSizesFifo : resources->hostMem->sizesFifo;
  int buffSize = llMode ? LL_BUFF_SIZE : ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  assert(args->substeps <= SIZES_FIFO_SIZE);

  uint64_t head = llMode ? resources->llStep : 0ULL;
  uint64_t tail = llMode ? resources->llStep : 0ULL;
  uint64_t end = head + args->nsteps;

  int idle = 0;
  void* requests[args->substeps];

  if (!args->needProxy) goto nextColl;

  // Update in case we skipped some collectives
  if (llMode == 0) resources->hostMem->opCount = args->opCount;

  while (head < end) {
    idle++;
    if (llMode) {
      if (tail < end && tail < head + args->substeps) {
        int slot = tail%args->substeps;
        int size = sizesFifo[slot];
        if (size != 0) {
          if (size == -1) size = 0;
          uint32_t flag = tail + 1;
          int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
          size = nFifoLines * sizeof(union ncclLLFifoLine);
          union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+slot*sliceSize);
          for (int i=0; i<nFifoLines; i++) {
            volatile uint32_t *f1 = &lines[i].flag1;
            volatile uint32_t *f2 = &lines[i].flag2;
            while (f1[0] != flag || f2[0] != flag);
          }
          NCCLCHECK(ncclNetIsend(resources->netSendComm, lines, size, ptrType, requests+slot));
          sizesFifo[slot] = size;
          tail++;
          idle = 0;
        }
      }
    } else while (tail < *prevTail) {
      // Send through network
      int slot = tail%args->substeps;
      NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+slot*sliceSize, sizesFifo[slot], ptrType, requests+slot));
      tail++;
      idle = 0;
    }
    if (head < tail) {
      int done;
      int slot = head%args->substeps;
      NCCLCHECK(ncclNetTest(requests[slot], &done, NULL));
      if (done) {
        if (llMode) sizesFifo[slot] = 0;
        head++;
        *prevHead = head;
        idle = 0;
      }
      if (idle) transportProxyIdle(idle);
    }
  }

  // Reset
  if (llMode == 0) *prevTail = 0;

nextColl:
  if (llMode == 0) resources->hostMem->opCount = args->opCount+1;
  else {
    resources->llStep += args->nsteps;
    // Don't forget to ack otherwise the GPU won't be able to push data.
    *prevHead = resources->llStep;
    if (resources->llStep > resources->llLastCleaning + LL_CLEAN_FREQ) {
      memset(localBuff, 0, LL_BUFF_SIZE);
      resources->llStep += NUM_LL_CHUNKS;
      *prevHead = resources->llStep;
      resources->llLastCleaning = resources->llStep;
    }
  }
  return ncclSuccess;
}

ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct netRecvResources* resources = (struct netRecvResources*) (ring->recv.transportResources);
  int llMode = args->llMode;

  volatile uint64_t* nextHead = llMode ? &resources->hostMem->llHead : &resources->hostMem->head;
  struct ncclSendRecvMem* localMem = resources->cudaSupport ? ring->devMem : resources->hostMem;
  char* localBuff = llMode ? localMem->llBuff : localMem->buff;
  char* nextBuff = (resources->cudaSupport == false && resources->hostDevMem) ? resources->hostDevMem->buff : NULL;
  int ptrType = resources->cudaSupport ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
  uint64_t* nextTail = resources->hostDevMem ? &resources->hostDevMem->tail : &resources->hostMem->tail;

  int buffSize = llMode ? LL_BUFF_SIZE : ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  uint64_t head = llMode ? resources->llStep : 0ULL;
  uint64_t tail = llMode ? resources->llStep : 0ULL;
  uint64_t end = head + args->nsteps;

  int idle = 0;
  void* requests[args->substeps];

  if (!args->needProxy) goto nextColl;

  if (llMode == 0) {
    // Waiting for next opCount is only needed before writing nextTail.
    uint64_t* nextOpCount = resources->hostDevMem ? &resources->hostDevMem->opCount : &resources->hostMem->opCount;
    transportProxyWait([=] { return *nextOpCount >= args->opCount; });
  }

  while (head < end) {
    idle++;
    while ((tail < head + args->substeps) && (tail < *nextHead + args->substeps) && (tail < end)) {
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
        if (llMode == 0) {
          if (ptrType == NCCL_PTR_CUDA) ncclNetFlush(resources->netRecvComm, localBuff+slot*sliceSize, size);
          *nextTail = head;
        }
      }
      idle = 0;
    }
    if (idle) transportProxyIdle(idle);
  }

  // Wait for last ack and reset
  if (llMode == 0) {
    transportProxyWait([=] { return *nextHead == head; });
    *nextHead = 0;
  }

nextColl:
  if (llMode) {
    resources->llStep += args->nsteps;
    if (resources->llStep > resources->llLastCleaning + LL_CLEAN_FREQ) {
      resources->llStep += NUM_LL_CHUNKS;
      while (*nextHead < resources->llStep);
      resources->llLastCleaning = resources->llStep;
    }
  }
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
