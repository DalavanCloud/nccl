/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "socket.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <poll.h>

static int numRequests = 0;
int* ncclSocketRequests = NULL;
int* ncclSocketRequestUsed = NULL;
pthread_mutex_t ncclSocketRequestsLock = PTHREAD_MUTEX_INITIALIZER;

int* ncclSocketGetRequest() {
  pthread_mutex_lock(&ncclSocketRequestsLock);
  for (int i=0; i<numRequests; i++) {
    if (ncclSocketRequestUsed[i] == 0) {
      ncclSocketRequestUsed[i] = 1; 
      pthread_mutex_unlock(&ncclSocketRequestsLock);
      return ncclSocketRequests + i;
    }
  }
  // No free request found, grow the pool
  int newNumRequests = numRequests + 32;
  int* newRequests = (int*)malloc(newNumRequests*sizeof(int));
  int* newUsed = (int*)malloc(newNumRequests*sizeof(int));
  for (int i=0; i<numRequests; i++) {
    newRequests[i] = ncclSocketRequests[i];
    newUsed[i] = ncclSocketRequestUsed[i];
  } 
  for (int i=numRequests; i<newNumRequests; i++)
    newUsed[i] = 0;
  free(ncclSocketRequests);
  ncclSocketRequests = newRequests;
  free(ncclSocketRequestUsed);
  ncclSocketRequestUsed = newUsed;
  numRequests = newNumRequests;
  pthread_mutex_unlock(&ncclSocketRequestsLock);
  return ncclSocketGetRequest();
}

void ncclSocketFreeRequest(int* request) {
  pthread_mutex_lock(&ncclSocketRequestsLock);
  ncclSocketRequestUsed[request-ncclSocketRequests] = 0;
  pthread_mutex_unlock(&ncclSocketRequestsLock);
}

struct ncclSocketHandle {
  struct socketAddress connect_addr;
};

struct ncclSocketRecvComm {
  char ifName[128];
  int nfds;
  int nfdsActive;
  struct pollfd* fds;
};

struct ncclSocketSendComm {
  int fd;
};

void ncclSocketAddFd(struct ncclSocketRecvComm* comm, int fd) {
  if (comm->nfdsActive >= comm->nfds) {
    // Grow the number of fds
    comm->nfds += 32;
    comm->fds = (struct pollfd*)realloc(comm->fds, (comm->nfds)*sizeof(struct pollfd));
  }
  comm->fds[comm->nfdsActive].fd = fd;
  comm->fds[comm->nfdsActive].events = POLLIN;
  comm->nfdsActive++;
}

int ncclSocketGetHandle(void* opaqueHandle, void** recvComm) {
  struct ncclSocketRecvComm* comm = (struct ncclSocketRecvComm*)malloc(sizeof(struct ncclSocketRecvComm));
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  assert(sizeof(struct ncclSocketHandle) < NCCL_NET_HANDLE_MAXSIZE);
  comm->nfds = comm->nfdsActive = 0;
  comm->fds = NULL;
  int listenfd;
  NCCLCHECK(createListenSocket(&listenfd, &handle->connect_addr.port));
  ncclSocketAddFd(comm, listenfd);
  NCCLCHECK(getIpAddr(&(handle->connect_addr.ip_addr), comm->ifName));
  *recvComm = comm;
  return 0;
}

int ncclSocketConnectHandle(void* opaqueHandle, void** sendComm) {
  struct ncclSocketSendComm* comm = (struct ncclSocketSendComm*)malloc(sizeof(struct ncclSocketSendComm));
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  NCCLCHECK(connectAddress(&handle->connect_addr, &comm->fd));
  *sendComm = comm;
  return 0;
};

int ncclSocketIsend(void* sendComm, void* data, int size, void** request) {
  struct ncclSocketSendComm* comm = (struct ncclSocketSendComm*)sendComm;
  *request = NULL;
  NCCLCHECK(socketSend(comm->fd, &size, sizeof(int)));
  NCCLCHECK(socketSend(comm->fd, data, size));
  return 0;
}

int ncclSocketIrecv(void* recvComm, void* data, int size, void** request) {
  struct ncclSocketRecvComm* comm = (struct ncclSocketRecvComm*)recvComm;
  poll(comm->fds, comm->nfdsActive, -1);
  while (comm->fds[0].revents) {
    // Listen socket : got a connection
    struct sockaddr_in sockaddr;
    socklen_t socklen = sizeof(struct sockaddr_in);
    int peerfd;
    SYSCHECKVAL(accept(comm->fds[0].fd, (struct sockaddr*)&sockaddr, &socklen), "accept", peerfd);
    ncclSocketAddFd(comm, peerfd);
    poll(comm->fds, comm->nfdsActive, -1);
  }
  for (int i=1; i<comm->nfdsActive; i++) {
    if (comm->fds[i].revents) {
      int recvSize;
      NCCLCHECK(socketReceive(comm->fds[i].fd, &recvSize, sizeof(int)));
      if (recvSize > size) {
        WARN("Message truncated : received %d bytes instead of %d\n", recvSize, size);
      }
      NCCLCHECK(socketReceive(comm->fds[i].fd, data, min(recvSize, size)));
      int* recvReq = ncclSocketGetRequest();
      *recvReq = recvSize;
      *request = recvReq;
      return 0;
    }
  }
  return 1;
}

int ncclSocketTest(void* request, int* done, int* size) {
  *done = 1;
  if (size && request) *size = *(int*)request;
  return 0;
}

int ncclSocketCloseSend(void* sendComm) {
  if (sendComm) {
    struct ncclSocketSendComm* comm = (struct ncclSocketSendComm*)sendComm;
    close(comm->fd);
    free(comm);
  }
  return 0;
}

int ncclSocketCloseRecv(void* recvComm) {
  if (recvComm) {
    struct ncclSocketRecvComm* comm = (struct ncclSocketRecvComm*)recvComm;
    for (int i=0; i<comm->nfdsActive; i++) close(comm->fds[i].fd);
    free(comm->fds);
    free(comm);
  }
  return 0;
}

ncclNet_t ncclNetSocket = {
  "Socket",
  ncclSocketGetHandle,
  ncclSocketConnectHandle,
  ncclSocketIsend,
  ncclSocketIrecv,
  ncclSocketTest,
  ncclSocketCloseSend,
  ncclSocketCloseRecv
};

extern "C" __attribute__ ((visibility("default")))
ncclNet_t* ncclNet = &ncclNetSocket;
