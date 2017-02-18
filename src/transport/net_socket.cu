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

#define MAX_IF_NAME_SIZE 16
#define MAX_IFS 16
static char ncclNetIfNames[MAX_IF_NAME_SIZE*MAX_IFS];
static struct in_addr ncclNetIfAddrs[MAX_IFS];
static int ncclNetIfs = -1;

static int searchDevices(const char* ifNamePrefix) {
  bool searchNot = (strlen(ifNamePrefix) > 0 && ifNamePrefix[0] == '^');
  if (searchNot) /* Skip the '^' */ ifNamePrefix++;
  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL || interface->ifa_addr->sa_family != AF_INET) continue;
    if (strncmp("lo", interface->ifa_name, strlen("lo")) == 0) continue; // Do not use loopback interfaces

    int matchLength = min((int)strlen(ifNamePrefix), MAX_IF_NAME_SIZE);
    int match = strncmp(interface->ifa_name, ifNamePrefix, matchLength);
    if ((match == 0) ^ searchNot) {
      // Store the interface name
      strcpy(ncclNetIfNames+ncclNetIfs*MAX_IF_NAME_SIZE, interface->ifa_name);
      // Store the IP address
      struct sockaddr_in* sa = (struct sockaddr_in*)(interface->ifa_addr);
      memcpy(ncclNetIfAddrs+ncclNetIfs, &sa->sin_addr, sizeof(struct sockaddr_in));
      INFO("NET/Socket : Using interface %s", interface->ifa_name);
      ncclNetIfs++;
    }
  }
  freeifaddrs(interfaces);
  return found;
}

static void initDevices() {
  if (ncclNetIfs == -1) {
    pthread_mutex_lock(&ncclSocketLock);
    if (ncclNetIfs == -1) {
      ncclNetIfs = 0;
      // User specified interface
      char* env = getenv("NCCL_SOCKET_IFNAME");
      if (env && strlen(env) > 1) searchDevices(env);
      if (ncclNetIfs == 0) searchDevices("ib");
      if (ncclNetIfs == 0) searchDevices("^lo");
      if (ncclNetIfs == 0) searchDevices("lo");
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
  int listenfd;
  NCCLCHECK(GetIpAddr(dev, &(handle->connect_addr.ip_addr)));
  NCCLCHECK(createListenSocket(&listenfd, handle->connect_addr.ip_addr, &handle->connect_addr.port));
  ncclSocketAddFd(comm, listenfd);
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
        return ncclInternalError;
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
  ncclSocketDevices,
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
