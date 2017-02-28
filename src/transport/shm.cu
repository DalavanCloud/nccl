/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "utils.h"
#include "transport.h"
#include "shm.h"
#include <unistd.h>
#include <cuda_runtime.h>

//#define SHM_PROXY

struct shmInfo {
  int rank;
  int cudaDev;
  int pid;
  uint64_t hostHash;
  int hostNumber;
};

struct shmSendConnectInfo {
  int pid;
  int id;
  int rank;
  int shmSize;
};

struct shmRecvConnectInfo {
#ifdef SHM_PROXY
  int direct;
  union {
    struct ncclSendRecvMem* directPtr;
    cudaIpcMemHandle_t devIpc;
  };
#else
  int pid;
  int id;
  int rank;
  int shmSize;
#endif
};

struct shmSendResources {
  int remShmSize;
  struct ncclSendRecvMem* remHostMem;
  struct ncclSendRecvMem* devRemHostMem;
  int shmSize;
  int* hostMem;
  int* devHostMem;
};

#define MAXSTEPS 8

struct shmRecvResources {
#ifdef SHM_PROXY
  int prevCudaDev;
  int localCudaDev;
  cudaStream_t prevStream;
  cudaStream_t localStream;
  cudaEvent_t syncEvent[MAXSTEPS];
  struct ncclSendRecvMem* remDevMem;
#else
  int remShmSize;
  struct ncclSendRecvMem* remHostMem;
  struct ncclSendRecvMem* devRemHostMem;
#endif
  int shmSize;
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
};

/* Fill information necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t shmFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  struct shmInfo* info = (struct shmInfo*)opaqueInfo;
  static_assert(sizeof(struct shmInfo) <= sizeof(ncclTinfo_t), "shm Info too large");
  info->rank = rank;
  CUDACHECK(cudaGetDevice(&info->cudaDev));
  info->pid = getpid();
  char hostname[1024];
  getHostName(hostname, 1024);
  info->hostHash=getHostHash(hostname);
  info->hostNumber=getHostNumber(hostname);
  return ncclSuccess;
}

/* Determine if we can communicate with the peer */
ncclResult_t shmCanConnect(int* ret, ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo) {
  static int shmDisabled = -1;
  if (shmDisabled == -1) {
    char* str = getenv("NCCL_SHM_DISABLE");
    shmDisabled = str ? atoi(str) : 0;
  }
  struct shmInfo* myInfo = (struct shmInfo*)myOpaqueInfo;
  struct shmInfo* peerInfo = (struct shmInfo*)peerOpaqueInfo;
  *ret = ((shmDisabled == 1) || (myInfo->hostHash != peerInfo->hostHash)) ? 0 : 1;
  return ncclSuccess;
}

static inline int groupFirst(int nranks, int* groups, int group) {
  for (int rank = 0; rank<nranks; rank++) {
    if (groups[rank] == group) return rank;
  }
  return -1;
}

static inline int groupLast(int nranks, int* groups, int group) {
  for (int rank = nranks-1; rank>=0; rank--) {
    if (groups[rank] == group) return rank;
  }
  return -1;
}

ncclResult_t shmGetRings(int nranks, int ngroups, int* groups, int* values, int* nringsRet, int* prev, int* next, int pattern) {
  if (pattern >= 1) {
    *nringsRet = 0;
    return ncclSuccess;
  }
  if (*nringsRet == MAXRINGS) *nringsRet = 1;
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
        source = groupLast(nranks, groups, group);
        destination = groupFirst(nranks, groups, nextGroup);
      } else {
        int prevGroup = (group-1+ngroups)%ngroups;
        destination = groupLast(nranks, groups, prevGroup);
        source = groupFirst(nranks, groups, group);
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

/* Create and return connect structures for this peer to connect to me */
ncclResult_t shmSendSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct shmInfo* myInfo = (struct shmInfo*)myOpaqueInfo;
  struct shmInfo* peerInfo = (struct shmInfo*)peerOpaqueInfo;

  struct shmSendResources* resources = (struct shmSendResources*)malloc(sizeof(struct shmSendResources));
  ring->send.transportResources = resources;

  struct shmRecvConnectInfo info;
#ifdef SHM_PROXY
  // Send devMem ptr to receiver so that proxy thread can update my head ptr
  if (myInfo->pid == peerInfo->pid) {
    info.direct = 1;
    info.directPtr = ring->devMem;
    INFO("SendSetup : sending devMem ptr %p", info.directPtr);
  } else {
    info.direct = 0;
    // Map IPC
    if (cudaIpcGetMemHandle(&info.devIpc, (void*)ring->devMem) != cudaSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d", myInfo->rank, peerInfo->cudaDev);
      return ncclInternalError;
    }
  }
  INFO("%d -> %d via proxy shared memory", myInfo->rank, peerInfo->rank);
#else
  char shmName[1024];
  sprintf(shmName, "nccl-shm-send-%d-%d-%d", myInfo->pid, ring->id, myInfo->rank);
  info.shmSize = resources->shmSize = sizeof(int);
  NCCLCHECK(shmOpen(shmName, resources->shmSize, (void**)&resources->hostMem, (void**)&resources->devHostMem, 1));
  
  INFO("%d -> %d via direct shared memory", myInfo->rank, peerInfo->rank);
  info.id = ring->id; info.rank = myInfo->rank; info.pid = myInfo->pid;
#endif
  static_assert(sizeof(struct shmRecvConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Recv Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmRecvConnectInfo));
  return ncclSuccess;
}

ncclResult_t shmRecvSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct shmInfo* myInfo = (struct shmInfo*)myOpaqueInfo;
  struct shmRecvResources* resources = (struct shmRecvResources*) malloc(sizeof(struct shmRecvResources));
  ring->recv.transportResources = resources;

  // Create streams for proxy
#ifdef SHM_PROXY
  struct shmInfo* peerInfo = (struct shmInfo*)peerOpaqueInfo;
  resources->prevCudaDev = peerInfo->cudaDev;
  CUDACHECK(cudaSetDevice(peerInfo->cudaDev));
  CUDACHECK(cudaStreamCreateWithFlags(&resources->prevStream, cudaStreamNonBlocking));
  resources->localCudaDev = myInfo->cudaDev;
  CUDACHECK(cudaSetDevice(myInfo->cudaDev));
  CUDACHECK(cudaStreamCreateWithFlags(&resources->localStream, cudaStreamNonBlocking));
  for (int i=0; i<MAXSTEPS; i++)
    CUDACHECK(cudaEventCreate(resources->syncEvent+i));
#endif

  struct shmSendConnectInfo info;

  char shmName[1024];
  sprintf(shmName, "nccl-shm-recv-%d-%d-%d", myInfo->pid, ring->id, myInfo->rank);
  info.shmSize = resources->shmSize = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  NCCLCHECK(shmOpen(shmName, resources->shmSize, (void**)&resources->hostMem, (void**)&resources->devHostMem, 1));
  
  info.id = ring->id; info.rank = myInfo->rank; info.pid = myInfo->pid;
  static_assert(sizeof(struct shmRecvConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Send Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmSendConnectInfo));
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t shmSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct shmSendConnectInfo* info = (struct shmSendConnectInfo*)connectInfo;
  struct shmSendResources* resources = (struct shmSendResources*)send->transportResources;

  char shmName[1024];
  sprintf(shmName, "nccl-shm-recv-%d-%d-%d", info->pid, info->id, info->rank);
  resources->remShmSize = info->shmSize;
  NCCLCHECK(shmOpen(shmName, resources->remShmSize, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, 0));
  // Remove the file to ensure proper clean-up
  NCCLCHECK(shmUnlink(shmName));

  send->transportResources = resources;
  send->conn.buff = resources->devRemHostMem->buff;
  send->conn.tail = &resources->devRemHostMem->tail;
  send->conn.opCount = &resources->devRemHostMem->opCount;
#ifndef SHM_PROXY
  send->conn.head = resources->devHostMem;
#endif
  return ncclSuccess;
}

ncclResult_t shmRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  struct shmRecvConnectInfo* info = (struct shmRecvConnectInfo*)connectInfo;

#ifdef SHM_PROXY
  // Setup receive proxy pointers
  if (info->direct) {
    INFO("ConnectRecv : using direct devMem ptr %p", info->directPtr);
    resources->remDevMem = info->directPtr;
  } else {
    CUDACHECK(cudaSetDevice(resources->prevCudaDev));
    CUDACHECK(cudaIpcOpenMemHandle((void**)&resources->remDevMem,
          info->devIpc, cudaIpcMemLazyEnablePeerAccess));
    CUDACHECK(cudaSetDevice(resources->localCudaDev));
  }
  recv->conn.head = &resources->devHostMem->head;
#else
  char shmName[1024];
  sprintf(shmName, "nccl-shm-send-%d-%d-%d", info->pid, info->id, info->rank);
  resources->remShmSize = info->shmSize;
  NCCLCHECK(shmOpen(shmName, resources->remShmSize, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, 0));
  NCCLCHECK(shmUnlink(shmName));
  recv->conn.head = &resources->devRemHostMem->head;
  recv->conn.buff = resources->devHostMem->buff;
  recv->conn.tail = &resources->devHostMem->tail;
  recv->conn.opCount = &resources->devHostMem->opCount;
#endif
  return ncclSuccess;
}

ncclResult_t shmSendFree(void* transportResources) {
  struct shmSendResources* resources = (struct shmSendResources*)transportResources;
  NCCLCHECK(shmClose(resources->hostMem, resources->devHostMem, resources->shmSize));
  NCCLCHECK(shmClose(resources->remHostMem, resources->devRemHostMem, resources->remShmSize));
  free(resources);
  return ncclSuccess;
}

ncclResult_t shmRecvFree(void* transportResources) {
  struct shmRecvResources* resources = (struct shmRecvResources*)transportResources;
  NCCLCHECK(shmClose(resources->hostMem, resources->devHostMem, resources->shmSize));
#ifdef SHM_PROXY
  CUDACHECK(cudaStreamDestroy(prevStream));
  CUDACHECK(cudaStreamDestroy(localStream));
  for (int i=0; i<MAXSTEPS; i++) {
    CUDACHECK(cudaEvenDestroy(resources->syncEvent[i]));
  }
  CUDACHECK(cudaIpcCloseMemHandle(resources->remDevMem));
#else
  NCCLCHECK(shmClose(resources->remHostMem, resources->devRemHostMem, resources->remShmSize));
#endif
  free(resources);
  return ncclSuccess;
}

#ifdef SHM_PROXY
ncclResult_t shmRecvProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct shmRecvResources* resources = (struct shmRecvResources*) (ring->recv.transportResources);
  struct ncclSendRecvMem* devMem = ring->devMem;
  volatile int* prevTail = &resources->hostMem->tail;
  int* prevHead = &resources->remDevMem->head;
  int* nextTail = &devMem->tail;
  int* nextOpCount = &devMem->opCount;
  volatile int* nextHead = &resources->hostMem->head;
  char* localBuff = resources->hostMem->buff;
  char* nextBuff = devMem->buff;
  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  // Update in case we skipped some collectives
  resources->hostMem->opCount = args->opCount;
  int val = 0;
  while (val != args->opCount) {
    CUDACHECK(cudaMemcpyAsync(&val, nextOpCount, sizeof(int), cudaMemcpyDeviceToHost, resources->localStream));
    CUDACHECK(cudaStreamSynchronize(resources->localStream));
  }
  int head = 0;
  int offset = 0;

  while (head < args->nsteps) {
    CUDACHECK(cudaSetDevice(resources->localCudaDev));
    transportProxyWait([=] { return head != *prevTail; });
    transportProxyWait([=] { return (head - *nextHead) < args->substeps; });
    head++;
    CUDACHECK(cudaMemcpyAsync(nextBuff+offset, localBuff+offset, sliceSize, cudaMemcpyHostToDevice, resources->localStream));
    CUDACHECK(cudaEventRecord(resources->syncEvent[head%args->substeps], resources->localStream));
    CUDACHECK(cudaMemcpyAsync(nextTail, &head, sizeof(int), cudaMemcpyHostToDevice, resources->localStream));

    CUDACHECK(cudaSetDevice(resources->prevCudaDev));
    CUDACHECK(cudaStreamWaitEvent(resources->prevStream, resources->syncEvent[head%args->substeps], 0));
    CUDACHECK(cudaMemcpyAsync(prevHead, &head, sizeof(int), cudaMemcpyHostToDevice, resources->prevStream));

    offset += sliceSize;
    if (offset == buffSize)
      offset = 0;
  }
  // Ensure all updates are pushed
  CUDACHECK(cudaSetDevice(resources->prevCudaDev));
  CUDACHECK(cudaStreamSynchronize(resources->prevStream));
  CUDACHECK(cudaSetDevice(resources->localCudaDev));
  CUDACHECK(cudaStreamSynchronize(resources->localStream));

  // Wait for last ack and reset
  transportProxyWait([=] { return *nextHead == head; });
  *nextHead = 0;
  *prevTail = 0;
  resources->hostMem->opCount = args->opCount+1;

  return ncclSuccess;
}
#endif

struct ncclTransport shmTransport = {
  "SHM",
  shmFillInfo,
  shmCanConnect,
  shmGetRings,
  { shmSendSetup, shmSendConnect, shmSendFree, NULL },
  { shmRecvSetup, shmRecvConnect, shmRecvFree,
#ifdef SHM_PROXY
    shmRecvProxy
#else
    NULL
#endif 
  }
};
