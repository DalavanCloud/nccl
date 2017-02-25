/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <poll.h>

int ncclSocketPtrSupport(int* supportedTypes) {
  *supportedTypes = NCCL_PTR_HOST;
  return 0;
}

static int numRequests = 0;
int* ncclSocketRequests = NULL;
int* ncclSocketRequestUsed = NULL;
pthread_mutex_t ncclSocketLock = PTHREAD_MUTEX_INITIALIZER;

int* ncclSocketGetRequest() {
  pthread_mutex_lock(&ncclSocketLock);
  for (int i=0; i<numRequests; i++) {
    if (ncclSocketRequestUsed[i] == 0) {
      ncclSocketRequestUsed[i] = 1; 
      pthread_mutex_unlock(&ncclSocketLock);
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
  pthread_mutex_unlock(&ncclSocketLock);
  return ncclSocketGetRequest();
}

void ncclSocketFreeRequest(int* request) {
  pthread_mutex_lock(&ncclSocketLock);
  ncclSocketRequestUsed[request-ncclSocketRequests] = 0;
  pthread_mutex_unlock(&ncclSocketLock);
}

struct ncclSocketHandle {
  struct socketAddress connectAddr;
};

struct ncclSocketRecvComm {
  char ifName[128];
  int listenFd;
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

#define MAX_IF_NAME_SIZE 16
#define MAX_IFS 16
static char ncclNetIfNames[MAX_IF_NAME_SIZE*MAX_IFS];
static struct in_addr ncclNetIfAddrs[MAX_IFS];
static int ncclNetIfs = -1;

static void initDevices() {
  if (ncclNetIfs == -1) {
    pthread_mutex_lock(&ncclSocketLock);
    if (ncclNetIfs == -1) {
      ncclNetIfs = 0;
      // User specified interface
      char* env = getenv("NCCL_SOCKET_IFNAME");
      if (env && strlen(env) > 1) {
        // Specified by user : find or fail
        ncclNetIfs = findInterfaces(env, ncclNetIfNames, ncclNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
      } else {
        // Try to automatically pick the right one
        // Start with IB
        ncclNetIfs = findInterfaces("ib", ncclNetIfNames, ncclNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
        // Then look for anything else (but not loopback)
        if (ncclNetIfs == 0) ncclNetIfs = findInterfaces("^lo", ncclNetIfNames, ncclNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
        // Don't try loopback. If we are we running intra-node we can always set env="lo".
        //if (ncclNetIfs == 0) ncclNetIfs = findInterfaces("lo", ncclNetIfNames, ncclNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
      }
      INFO("NET/Socket : %d interfaces found", ncclNetIfs);
    }
    pthread_mutex_unlock(&ncclSocketLock);
  }
}

int ncclSocketDevices(int* ndev, int** distances) {
  initDevices();
  *ndev = ncclNetIfs;
  int* dists = (int*)malloc(ncclNetIfs*sizeof(int));
  for (int i=0; i<ncclNetIfs; i++) dists[i] = 0;
  *distances = dists;
  return ncclSuccess;
}

static ncclResult_t GetIpAddr(int dev, struct in_addr* addr) {
  if (ncclNetIfs == -1) initDevices();
  if (dev > ncclNetIfs) return ncclInternalError;
  memcpy(addr, ncclNetIfAddrs+dev, sizeof(struct in_addr));
  return ncclSuccess;
}

int ncclSocketGetHandle(int dev, void* opaqueHandle, void** recvComm) {
  struct ncclSocketRecvComm* comm = (struct ncclSocketRecvComm*)malloc(sizeof(struct ncclSocketRecvComm));
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  assert(sizeof(struct ncclSocketHandle) < NCCL_NET_HANDLE_MAXSIZE);
  comm->nfds = comm->nfdsActive = 0;
  comm->fds = NULL;
  NCCLCHECK(GetIpAddr(dev, &(handle->connectAddr.ip_addr)));
  NCCLCHECK(createListenSocket(&comm->listenFd, handle->connectAddr.ip_addr, &handle->connectAddr.port));
  *recvComm = comm;
  return 0;
}

int ncclSocketConnectHandle(int dev, void* opaqueHandle, void** sendComm) {
  struct ncclSocketSendComm* comm = (struct ncclSocketSendComm*)malloc(sizeof(struct ncclSocketSendComm));
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  NCCLCHECK(connectAddress(&handle->connectAddr, ncclNetIfAddrs[dev], &comm->fd));
  *sendComm = comm;
  return 0;
}

int ncclSocketAccept(void* recvComm) {
  struct ncclSocketRecvComm* comm = (struct ncclSocketRecvComm*)recvComm;
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  int peerfd;
  SYSCHECKVAL(accept(comm->listenFd, (struct sockaddr*)&sockaddr, &socklen), "accept", peerfd);
  ncclSocketAddFd(comm, peerfd);
  return 0;
}

int ncclSocketIsend(void* sendComm, void* data, int size, int type, void** request) {
  if (type != NCCL_PTR_HOST) return 1;
  struct ncclSocketSendComm* comm = (struct ncclSocketSendComm*)sendComm;
  *request = NULL;
  NCCLCHECK(socketSend(comm->fd, &size, sizeof(int)));
  NCCLCHECK(socketSend(comm->fd, data, size));
  return 0;
}

int ncclSocketIrecv(void* recvComm, void* data, int size, int type, void** request) {
  if (type != NCCL_PTR_HOST) return 1;
  struct ncclSocketRecvComm* comm = (struct ncclSocketRecvComm*)recvComm;
  while (1) {
    poll(comm->fds, comm->nfdsActive, -1);
    for (int i=0; i<comm->nfdsActive; i++) {
      if (comm->fds[i].revents) {
        int recvSize;
        NCCLCHECK(socketReceive(comm->fds[i].fd, &recvSize, sizeof(int)));
        if (recvSize > size) {
          WARN("Message truncated : received %d bytes instead of %d\n", recvSize, size);
          return ncclInternalError;
        }
        NCCLCHECK(socketReceive(comm->fds[i].fd, data, min(recvSize, size)));
        int* recvReq = ncclSocketGetRequest();
        *recvReq = recvSize;
        *request = recvReq;
        return 0;
      }
    }
  }
}

int ncclSocketTest(void* request, int* done, int* size) {
  int *r = (int*)request;
  *done = 1;
  if (r) {
    if (size) *size = *r;
    ncclSocketFreeRequest(r);
  }
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
  ncclSocketPtrSupport,
  ncclSocketDevices,
  ncclSocketGetHandle,
  ncclSocketConnectHandle,
  ncclSocketAccept,
  ncclSocketIsend,
  ncclSocketIrecv,
  ncclSocketTest,
  ncclSocketCloseSend,
  ncclSocketCloseRecv
};
