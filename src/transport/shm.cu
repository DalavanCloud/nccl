#include "core.h"
#include "utils.h"
#include "transport.h"
#include <unistd.h>
#include <cuda_runtime.h>

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
  int direct;
  union {
    struct ncclSendRecvMem* directPtr;
    cudaIpcMemHandle_t devIpc;
  };
};

struct shmResourcesSend {
  int fd;
  struct ncclSendRecvMem* remHostMem;
};

struct shmResourcesRecv {
  int fd;
  int prevCudaDev;
  int localCudaDev;
  cudaStream_t prevStream;
  cudaStream_t localStream;
  cudaEvent_t syncEvent;
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
  struct ncclSendRecvMem* remDevMem;
};

/* Fill infomation necessary to exchange between ranks to choose whether or not
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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t shmSetupSend(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring, int* select) {
  struct shmInfo* myInfo = (struct shmInfo*)myOpaqueInfo;
  struct shmInfo* peerInfo = (struct shmInfo*)peerOpaqueInfo;
  if (myInfo->hostHash != peerInfo->hostHash) {
    *select = 0;
    return ncclSuccess;
  }

  struct shmConnectRecvInfo info;

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
      // We could return an error, but maybe it is better to gracefully disable
      // p2p and fall back on something else.
      *select = 0;
      return ncclSuccess;
    }
  }
  INFO("%d [%d] -> %d [%d] via shared memory", myInfo->rank, myInfo->cudaDev, peerInfo->rank, peerInfo->cudaDev);
  static_assert(sizeof(struct shmConnectRecvInfo) <= sizeof(struct ncclConnect), "shm Connect Recv Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmConnectRecvInfo));
  *select = 1;
  return ncclSuccess;
}

ncclResult_t shmSetupRecv(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring, int* select) {
  struct shmInfo* myInfo = (struct shmInfo*)myOpaqueInfo;
  struct shmInfo* peerInfo = (struct shmInfo*)peerOpaqueInfo;
  if (myInfo->hostHash != peerInfo->hostHash) {
    *select = 0;
    return ncclSuccess;
  }

  struct shmResourcesRecv* resources = (struct shmResourcesRecv*) malloc(sizeof(struct shmResourcesRecv));
  ring->recv.transportResources = resources;

  // Create streams for proxy
  resources->prevCudaDev = peerInfo->cudaDev;
  CUDACHECK(cudaSetDevice(peerInfo->cudaDev));
  CUDACHECK(cudaStreamCreateWithFlags(&resources->prevStream, cudaStreamNonBlocking));
  resources->localCudaDev = myInfo->cudaDev;
  CUDACHECK(cudaSetDevice(myInfo->cudaDev));
  CUDACHECK(cudaStreamCreateWithFlags(&resources->localStream, cudaStreamNonBlocking));
  CUDACHECK(cudaEventCreate(&resources->syncEvent));

  char shmname[1024];
  sprintf(shmname, "nccl-%d-%d-%d", myInfo->pid, ring->id, ring->rank);
  int shmsize = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;

  int fd = shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    WARN("shm_open failed to open %s", shmname);
    *select = 0;
    return ncclSuccess;
  }

  if (ftruncate(fd, shmsize) == -1) {
    WARN("ftruncate failed to allocate %ld bytes", shmsize);
    shm_unlink(shmname);
    close(fd);
    *select = 0;
    return ncclSuccess;
  }

  void *ptr = (struct ncclSendRecvMem*)mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    WARN("failure in mmap");
    shm_unlink(shmname);
    close(fd);
    *select = 0;
    return ncclSuccess;
  }
  close(fd);

  if (cudaHostRegister(ptr, shmsize, cudaHostRegisterMapped) != cudaSuccess) {
    WARN("failed to register host buffer");
    shm_unlink(shmname);
    munmap(ptr, shmsize);
    *select = 0;
    return ncclSuccess;
  }   

  if (cudaHostGetDevicePointer(&resources->devHostMem, ptr, 0) != cudaSuccess) {
    WARN("failed to get device pointer for local shmem");
    shm_unlink(shmname);
    munmap(resources->hostMem, shmsize);
    *select = 0;
    return ncclSuccess;
  }
  resources->hostMem = (struct ncclSendRecvMem*)ptr;
  
  struct shmConnectSendInfo info;
  info.id = ring->id; info.rank = ring->rank; info.pid = myInfo->pid; info.shmsize = shmsize;
  static_assert(sizeof(struct shmConnectRecvInfo) <= sizeof(struct ncclConnect), "shm Connect Send Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmConnectSendInfo));
  *select = 1;
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t shmConnectRecv(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct shmResourcesRecv* resources = (struct shmResourcesRecv*)recv->transportResources;
  recv->conn.head = &resources->devHostMem->head;

  // Setup receive proxy pointers
  struct shmConnectRecvInfo* info = (struct shmConnectRecvInfo*)connectInfo;
  if (info->direct) {
    INFO("ConnectRecv : using direct devMem ptr %p", info->directPtr);
    resources->remDevMem = info->directPtr;
  } else {
    CUDACHECK(cudaSetDevice(resources->prevCudaDev));
    CUDACHECK(cudaIpcOpenMemHandle((void**)resources->remDevMem,
          info->devIpc, cudaIpcMemLazyEnablePeerAccess));
  }
  return ncclSuccess;
}

ncclResult_t shmConnectSend(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct shmConnectSendInfo* info = (struct shmConnectSendInfo*)connectInfo;
  struct shmResourcesSend* resources = (struct shmResourcesSend*)malloc(sizeof(struct shmResourcesSend));

  char shmname[1024];
  sprintf(shmname, "nccl-%d-%d-%d", info->pid, info->id, info->rank);
  int shmsize = info->shmsize;

  int fd = shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    WARN("shm_open failed to open %s", shmname);
    return ncclInternalError;
  }

  void *ptr = (struct ncclSendRecvMem*)mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    WARN("failure in mmap");
    shm_unlink(shmname);
    close(fd);
    return ncclInternalError;
  }
  close(fd);

  if (cudaHostRegister(ptr, shmsize, cudaHostRegisterMapped) != cudaSuccess) {
    WARN("failed to register host buffer");
    shm_unlink(shmname);
    munmap(ptr, shmsize);
    return ncclInternalError;
  }   

  struct ncclSendRecvMem* remHostMem;
  if (cudaHostGetDevicePointer(&remHostMem, ptr, 0) != cudaSuccess) {
    WARN("failed to get device pointer for local shmem");
    shm_unlink(shmname);
    munmap(ptr, shmsize);
    return ncclInternalError;
  }

  resources->remHostMem = remHostMem;
  send->transportResources = resources;
  send->conn.buff = remHostMem->buff;
  send->conn.tail = &remHostMem->tail;
  return ncclSuccess;
}

ncclResult_t shmRecvProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct shmResourcesRecv* resources = (struct shmResourcesRecv*) (ring->recv.transportResources);
  struct ncclSendRecvMem* devMem = ring->devMem;
  volatile int* prevTail = &resources->hostMem->tail;
  int* prevHead = &resources->remDevMem->head;
  int* nextTail = &devMem->tail;
  volatile int* nextHead = &resources->hostMem->head;
  char* localBuff = resources->hostMem->buff;
  char* nextBuff = devMem->buff;
  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  int val = 1;
  CUDACHECK(cudaSetDevice(resources->prevCudaDev));
  while (val != 0) {
    CUDACHECK(cudaMemcpyAsync(&val, prevHead, sizeof(int), cudaMemcpyDeviceToHost, resources->prevStream));
    CUDACHECK(cudaStreamSynchronize(resources->prevStream));
  }
  CUDACHECK(cudaSetDevice(resources->localCudaDev));
  while (val != 0) {
    CUDACHECK(cudaMemcpyAsync(&val, nextTail, sizeof(int), cudaMemcpyDeviceToHost, resources->localStream));
    CUDACHECK(cudaStreamSynchronize(resources->localStream));
  }
  int head = 0;
  int offset = 0;

  while (head < args->nsteps) {
    CUDACHECK(cudaSetDevice(resources->localCudaDev));
    while ((head - *prevTail) == 0);
    while ((head - *nextHead) >= args->substeps);
    head++;
    //printf("Proxy : %d : copying from/to %X\n", head, offset);
    CUDACHECK(cudaMemcpyAsync(nextBuff+offset, localBuff+offset, sliceSize, cudaMemcpyHostToDevice, resources->localStream));
    CUDACHECK(cudaMemcpyAsync(nextTail, &head, sizeof(int), cudaMemcpyHostToDevice, resources->localStream));
    CUDACHECK(cudaEventRecord(resources->syncEvent, resources->localStream));

    //CUDACHECK(cudaSetDevice(resources->prevCudaDev));
    CUDACHECK(cudaStreamWaitEvent(resources->prevStream, resources->syncEvent, 0));
    CUDACHECK(cudaMemcpyAsync(prevHead, &head, sizeof(int), cudaMemcpyHostToDevice, resources->prevStream));

    offset += sliceSize;
    if (offset == buffSize)
      offset = 0;
  }
  // Ensure all updates are pushed
  CUDACHECK(cudaSetDevice(resources->localCudaDev));
  CUDACHECK(cudaStreamSynchronize(resources->localStream));
  CUDACHECK(cudaSetDevice(resources->prevCudaDev));
  CUDACHECK(cudaStreamSynchronize(resources->prevStream));

  // Wait for last ack and reset
//  printf("Flags at end : %d | %d %d\n", head, *nextHead, *prevTail);
  *prevTail = 0;
  while (*nextHead < head);
  *nextHead = 0;

  return ncclSuccess;
}

struct ncclTransport shmTransport = {
  shmFillInfo,
  { shmSetupSend, shmConnectSend, NULL },
  { shmSetupRecv, shmConnectRecv, shmRecvProxy }
};
