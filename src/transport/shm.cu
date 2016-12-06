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

struct shmConnectSendInfo {
  int pid;
  int id;
  int rank;
  int shmsize;
};

struct shmConnectRecvInfo {
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
  int shmsize;
#endif
};

struct shmResourcesSend {
  struct ncclSendRecvMem* remHostMem;
  struct ncclSendRecvMem* devRemHostMem;
  int* hostMem;
  int* devHostMem;
};

#define MAXSTEPS 8

struct shmResourcesRecv {
#ifdef SHM_PROXY
  int prevCudaDev;
  int localCudaDev;
  cudaStream_t prevStream;
  cudaStream_t localStream;
  cudaEvent_t syncEvent[MAXSTEPS];
  struct ncclSendRecvMem* remDevMem;
#else
  struct ncclSendRecvMem* remHostMem;
  struct ncclSendRecvMem* devRemHostMem;
#endif
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
  struct shmInfo* myInfo = (struct shmInfo*)myOpaqueInfo;
  struct shmInfo* peerInfo = (struct shmInfo*)peerOpaqueInfo;
  *ret = myInfo->hostHash != peerInfo->hostHash ? 0 : 1;
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t shmSetupSend(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct shmInfo* myInfo = (struct shmInfo*)myOpaqueInfo;
  struct shmInfo* peerInfo = (struct shmInfo*)peerOpaqueInfo;

  struct shmResourcesSend* resources = (struct shmResourcesSend*)malloc(sizeof(struct shmResourcesSend));
  ring->send.transportResources = resources;

  struct shmConnectRecvInfo info;
#ifdef SHM_PROXY
  // Send devMem ptr to receiver so that proxy thread can update my head ptr
  if (myInfo->pid == peerInfo->pid) {
    info.direct = 1;
    info.directPtr = ring->devMem;
    INFO("SetupSend : sending devMem ptr %p", info.directPtr);
  } else {
    info.direct = 0;
    // Map IPC
    if (cudaIpcGetMemHandle(&info.devIpc, (void*)ring->devMem) != cudaSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d", ring->rank, peerInfo->cudaDev);
      return ncclInternalError;
    }
  }
  INFO("%d -> %d via proxy shared memory", myInfo->rank, peerInfo->rank);
#else
  char shmname[1024];
  sprintf(shmname, "nccl-shm-send-%d-%d-%d", myInfo->pid, ring->id, ring->rank);
  NCCLCHECK(shmOpen(shmname, sizeof(int), (void**)&resources->hostMem, (void**)&resources->devHostMem, 1));
  
  INFO("%d -> %d via direct shared memory", myInfo->rank, peerInfo->rank);
  info.id = ring->id; info.rank = ring->rank; info.pid = myInfo->pid; info.shmsize = sizeof(int);
#endif
  static_assert(sizeof(struct shmConnectRecvInfo) <= sizeof(struct ncclConnect), "shm Connect Recv Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmConnectRecvInfo));
  return ncclSuccess;
}

ncclResult_t shmSetupRecv(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct shmInfo* myInfo = (struct shmInfo*)myOpaqueInfo;
  struct shmResourcesRecv* resources = (struct shmResourcesRecv*) malloc(sizeof(struct shmResourcesRecv));
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

  char shmname[1024];
  sprintf(shmname, "nccl-shm-recv-%d-%d-%d", myInfo->pid, ring->id, ring->rank);
  int shmsize = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  NCCLCHECK(shmOpen(shmname, shmsize, (void**)&resources->hostMem, (void**)&resources->devHostMem, 1));
  
  struct shmConnectSendInfo info;
  info.id = ring->id; info.rank = ring->rank; info.pid = myInfo->pid; info.shmsize = shmsize;
  static_assert(sizeof(struct shmConnectRecvInfo) <= sizeof(struct ncclConnect), "shm Connect Send Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmConnectSendInfo));
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t shmConnectRecv(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct shmResourcesRecv* resources = (struct shmResourcesRecv*)recv->transportResources;
  struct shmConnectRecvInfo* info = (struct shmConnectRecvInfo*)connectInfo;

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
  char shmname[1024];
  sprintf(shmname, "nccl-shm-send-%d-%d-%d", info->pid, info->id, info->rank);
  NCCLCHECK(shmOpen(shmname, info->shmsize, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, 0));
  recv->conn.head = &resources->devRemHostMem->head;
  recv->conn.buff = resources->devHostMem->buff;
  recv->conn.tail = &resources->devHostMem->tail;
  recv->conn.opCount = &resources->devHostMem->opCount;
#endif
  return ncclSuccess;
}

ncclResult_t shmConnectSend(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct shmConnectSendInfo* info = (struct shmConnectSendInfo*)connectInfo;
  struct shmResourcesSend* resources = (struct shmResourcesSend*)send->transportResources;

  char shmname[1024];
  sprintf(shmname, "nccl-shm-recv-%d-%d-%d", info->pid, info->id, info->rank);
  NCCLCHECK(shmOpen(shmname, info->shmsize, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, 0));

  send->transportResources = resources;
  send->conn.buff = resources->devRemHostMem->buff;
  send->conn.tail = &resources->devRemHostMem->tail;
  send->conn.opCount = &resources->devRemHostMem->opCount;
#ifndef SHM_PROXY
  send->conn.head = resources->devHostMem;
#endif
  return ncclSuccess;
}

#ifdef SHM_PROXY
ncclResult_t shmRecvProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct shmResourcesRecv* resources = (struct shmResourcesRecv*) (ring->recv.transportResources);
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
  shmFillInfo,
  shmCanConnect,
  { shmSetupSend, shmConnectSend, NULL },
  { shmSetupRecv, shmConnectRecv, 
#ifdef SHM_PROXY
    shmRecvProxy
#else
    NULL
#endif 
  }
};
