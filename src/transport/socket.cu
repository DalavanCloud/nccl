/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "transport.h"
#include <cuda_runtime.h>
#include "socket.h"
#include <assert.h>

struct socketInfo {
  int rank;
  int listen_fd;
  struct socketAddress connect_addr;
};

struct socketSendResources {
  int fd;
  cudaStream_t stream;
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
};

#define MAXSTEPS 8

struct socketRecvResources {
  int listen_fd;
  int fd;
  cudaStream_t stream;
  cudaEvent_t syncEvent[MAXSTEPS];
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
};

/* Fill information necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t socketFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  struct socketInfo* info = (struct socketInfo*)opaqueInfo;
  static_assert(sizeof(struct socketInfo) <= sizeof(ncclTinfo_t), "socket Info too large");
  info->rank = rank;
  info->listen_fd = -1;
  return ncclSuccess;
}

ncclResult_t socketCreateListen(struct socketInfo* info, char* ifname) {
  if (info->listen_fd == -1) {
    NCCLCHECK(createListenSocket(&info->listen_fd, &info->connect_addr.port));
    NCCLCHECK(getIpAddr(&(info->connect_addr.ip_addr), ifname));
  }
  return ncclSuccess;
}

/* Determine if we can communicate with the peer */
ncclResult_t socketCanConnect(int* ret, ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo) {
  *ret = 1;
  return ncclSuccess;
}

ncclResult_t socketGetRings(int nranks, int ngroups, int* groups, int* values, int* nringsRet, int* prev, int* next, int pattern) {
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
      int source = groupLast(nranks, groups, group);
      int destination = groupFirst(nranks, groups, nextGroup);
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

/* Create and return connect structures for this peer to connect to me */
ncclResult_t socketSendSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct socketSendResources* resources = (struct socketSendResources*) malloc(sizeof(struct socketSendResources));
  ring->send.transportResources = resources;

  // Create stream for proxy
  CUDACHECK(cudaStreamCreateWithFlags(&resources->stream, cudaStreamNonBlocking));

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  CUDACHECK(cudaHostAlloc(&resources->hostMem, size, cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));

  // Just pass the socket info through
  static_assert(sizeof(struct socketInfo) <= sizeof(struct ncclConnect), "socket Connect Info is too big");
  memcpy(connectInfo, myOpaqueInfo, sizeof(struct socketInfo));
  return ncclSuccess;
}

ncclResult_t socketRecvSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct socketRecvResources* resources = (struct socketRecvResources*) malloc(sizeof(struct socketRecvResources));
  ring->recv.transportResources = resources;

  // Create stream for proxy
  CUDACHECK(cudaStreamCreateWithFlags(&resources->stream, cudaStreamNonBlocking));
  // And event
  for (int i=0; i<MAXSTEPS; i++)
    CUDACHECK(cudaEventCreate(resources->syncEvent+i));

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  CUDACHECK(cudaHostAlloc(&resources->hostMem, size, cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));
  
  // Just pass the socket info through
  struct socketInfo* myInfo = (struct socketInfo*)myOpaqueInfo;
  char ifname[128];
  NCCLCHECK(socketCreateListen(myInfo, ifname));
  resources->listen_fd = myInfo->listen_fd; 
  struct socketInfo* peerInfo = (struct socketInfo*)peerOpaqueInfo;
  INFO("%d -> %d via TCP/%s", peerInfo->rank, myInfo->rank, ifname);
  memcpy(connectInfo, myOpaqueInfo, sizeof(struct socketInfo));
  return ncclSuccess;
}

ncclResult_t socketSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct socketSendResources* resources = (struct socketSendResources*)send->transportResources;
  send->conn.buff = resources->devHostMem->buff;
  send->conn.tail = &resources->devHostMem->tail;
  send->conn.opCount = &resources->devHostMem->opCount;
  send->conn.fifo = resources->devHostMem->sizesFifo;

  // Setup receive proxy socket/pointers
  struct socketInfo* info = (struct socketInfo*)connectInfo;
  NCCLCHECK(connectAddress(&info->connect_addr, &resources->fd));
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t socketRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct socketRecvResources* resources = (struct socketRecvResources*)recv->transportResources;
  recv->conn.head = &resources->devHostMem->head;

  // We will finish the socket setup at beginning of Recv proxy
  resources->fd = 0;
  return ncclSuccess;
}

ncclResult_t socketSendFree(void* transportResources) {
  struct socketSendResources* resources = (struct socketSendResources*)transportResources;
  SYSCHECK(close(resources->fd), "close");
  CUDACHECK(cudaStreamDestroy(resources->stream));
  CUDACHECK(cudaFreeHost(resources->hostMem));
  free(resources);
  return ncclSuccess;
}

ncclResult_t socketRecvFree(void* transportResources) {
  struct socketRecvResources* resources = (struct socketRecvResources*)transportResources;
  SYSCHECK(close(resources->listen_fd), "close");
  SYSCHECK(close(resources->fd), "close");
  CUDACHECK(cudaStreamDestroy(resources->stream));
  for (int i=0; i<MAXSTEPS; i++) {
    CUDACHECK(cudaEventDestroy(resources->syncEvent[i]));
  }
  CUDACHECK(cudaFreeHost(resources->hostMem));
  free(resources);
  return ncclSuccess;
}

ncclResult_t socketSendProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct socketSendResources* resources = (struct socketSendResources*) (ring->send.transportResources);
  struct ncclSendRecvMem* devMem = ring->devMem;
  volatile int* prevTail = &resources->hostMem->tail;
  int* prevHead = &devMem->head;
  char* localBuff = resources->hostMem->buff;
  int* sizesFifo = resources->hostMem->sizesFifo;
  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  int head = 0;
  int offset = 0;

  // Update in case we skipped some collectives
  resources->hostMem->opCount = args->opCount;

  while (head < args->nsteps) {
    // Receive from GPU
    transportProxyWait([=] { return head != *prevTail; });

    // Send to socket
    int size = sizesFifo[head%args->substeps];
    NCCLCHECK(socketSend(resources->fd, &size, sizeof(size)));
    NCCLCHECK(socketSend(resources->fd, localBuff+offset, size));
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

ncclResult_t socketRecvProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct socketRecvResources* resources = (struct socketRecvResources*) (ring->recv.transportResources);
  struct ncclSendRecvMem* devMem = ring->devMem;
  int* nextTail = &devMem->tail;
  int* nextOpCount = &devMem->opCount;
  volatile int* nextHead = &resources->hostMem->head;
  char* localBuff = resources->hostMem->buff;
  char* nextBuff = devMem->buff;
  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;
  assert(MAXSTEPS >= args->substeps);

  if (resources->fd == 0) {
    struct sockaddr_in sockaddr;
    socklen_t socklen = sizeof(struct sockaddr_in);
    SYSCHECKVAL(accept(resources->listen_fd, (struct sockaddr*)&sockaddr, &socklen), "accept", resources->fd);
  }

  int val = 0;
  while (val != args->opCount) {
    CUDACHECK(cudaMemcpyAsync(&val, nextOpCount, sizeof(int), cudaMemcpyDeviceToHost, resources->stream));
    CUDACHECK(cudaStreamSynchronize(resources->stream));
  }
  int head = 0;
  int offset = 0;

  while (head < args->nsteps) {
    // Receive from socket
    CUDACHECK(cudaEventSynchronize(resources->syncEvent[head%args->substeps]));
    int size;
    NCCLCHECK(socketReceive(resources->fd, &size, sizeof(size)));
    NCCLCHECK(socketReceive(resources->fd, localBuff+offset, size));

    // Send to GPU
    transportProxyWait([=] { return (head - *nextHead) < args->substeps; });
    CUDACHECK(cudaMemcpyAsync(nextBuff+offset, localBuff+offset, size, cudaMemcpyHostToDevice, resources->stream));
    CUDACHECK(cudaEventRecord(resources->syncEvent[head%args->substeps], resources->stream));
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

struct ncclTransport socketTransport = {
  "TCP",
  socketFillInfo,
  socketCanConnect,
  socketGetRings,
  { socketSendSetup, socketSendConnect, socketSendFree, socketSendProxy },
  { socketRecvSetup, socketRecvConnect, socketRecvFree, socketRecvProxy }
};
