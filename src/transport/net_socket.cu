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

/* Init functions */

int ncclSocketPtrSupport(int dev, int* supportedTypes) {
  *supportedTypes = NCCL_PTR_HOST;
  return 0;
}

#define MAX_IF_NAME_SIZE 16
#define MAX_IFS 16
static char ncclNetIfNames[MAX_IF_NAME_SIZE*MAX_IFS];
static union socketAddress ncclNetIfAddrs[MAX_IFS];
static int ncclNetIfs = -1;
pthread_mutex_t ncclSocketLock = PTHREAD_MUTEX_INITIALIZER;

static void initDevices() {
  if (ncclNetIfs == -1) {
    pthread_mutex_lock(&ncclSocketLock);
    if (ncclNetIfs == -1) {
      // Allow user to force the INET socket family selection
      int family = envSocketFamily();
      ncclNetIfs = 0;
      // User specified interface
      char* env = getenv("NCCL_SOCKET_IFNAME");
      if (env && strlen(env) > 1) {
        // Specified by user : find or fail
        ncclNetIfs = findInterfaces(env, ncclNetIfNames, ncclNetIfAddrs, family, MAX_IF_NAME_SIZE, MAX_IFS);
      } else {
        // Try to automatically pick the right one
        // Start with IB
        ncclNetIfs = findInterfaces("ib", ncclNetIfNames, ncclNetIfAddrs, family, MAX_IF_NAME_SIZE, MAX_IFS);
        // Then look for anything else (but not loopback)
        if (ncclNetIfs == 0) ncclNetIfs = findInterfaces("^lo", ncclNetIfNames, ncclNetIfAddrs, family, MAX_IF_NAME_SIZE, MAX_IFS);
        // Don't try loopback. If we are we running intra-node we can always set env="lo".
        //if (ncclNetIfs == 0) ncclNetIfs = findInterfaces("lo", ncclNetIfNames, ncclNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
      }
      INFO("NET/Socket : %d interfaces found", ncclNetIfs);
    }
    pthread_mutex_unlock(&ncclSocketLock);
  }
}

int ncclSocketDevices(int* ndev, int** scores) {
  initDevices();
  *ndev = ncclNetIfs;
  int* sc = (int*)malloc(ncclNetIfs*sizeof(int));
  for (int i=0; i<ncclNetIfs; i++) sc[i] = 1;
  *scores = sc;
  return ncclSuccess;
}

static ncclResult_t GetSocketAddr(int dev, union socketAddress* addr) {
  if (ncclNetIfs == -1) initDevices();
  if (dev > ncclNetIfs) return ncclInternalError;
  memcpy(addr, ncclNetIfAddrs+dev, sizeof(*addr));
  return ncclSuccess;
}

/* Communication functions */

struct ncclSocketHandle {
  union socketAddress connectAddr;
};

struct ncclSocketRequest {
  int used;
  int size;
};

struct ncclSocketReqs {
  int nreqs;
  struct ncclSocketRequest* requests;
};

struct ncclSocketComm {
  int fd;
  struct ncclSocketReqs reqs;
};

struct ncclSocketComm* ncclSocketNewComm() {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)malloc(sizeof(struct ncclSocketComm));
  comm->reqs.nreqs = 0;
  comm->reqs.requests = NULL;
  comm->fd = -1;
  return comm;
}

int ncclSocketListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclSocketComm* comm = ncclSocketNewComm();
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclSocketHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclSocketHandle size too large");
  NCCLCHECK(GetSocketAddr(dev, &(handle->connectAddr)));
  NCCLCHECK(createListenSocket(&comm->fd, &handle->connectAddr));
  *listenComm = comm;
  return 0;
}

int ncclSocketConnect(int dev, void* opaqueHandle, void** sendComm) {
  struct ncclSocketComm* comm = ncclSocketNewComm();
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  NCCLCHECK(connectAddress(&handle->connectAddr, &ncclNetIfAddrs[dev], &comm->fd));
  *sendComm = comm;
  return 0;
}

int ncclSocketAccept(void* listenComm, void** recvComm) {
  struct ncclSocketComm* lComm = (struct ncclSocketComm*)listenComm;
  struct ncclSocketComm* rComm = ncclSocketNewComm();
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", rComm->fd);
  *recvComm = rComm;
  return 0;
}

struct ncclSocketRequest* ncclSocketGetRequest(struct ncclSocketReqs* reqs) {
  for (int i=0; i<reqs->nreqs; i++) {
    if (reqs->requests[i].used == 0) {
      reqs->requests[i].used = 1; 
      return reqs->requests + i;
    }
  }
  // No free request found, grow the pool
  int newNumRequests = reqs->nreqs + 32;
  reqs->requests = (struct ncclSocketRequest*)realloc(reqs->requests, newNumRequests*sizeof(struct ncclSocketRequest));
  for (int i=reqs->nreqs; i<newNumRequests; i++)
    reqs->requests[i].used = 0;
  reqs->nreqs = newNumRequests;
  return ncclSocketGetRequest(reqs);
}

int ncclSocketIsend(void* sendComm, void* data, int size, int type, void** request) {
  if (type != NCCL_PTR_HOST) return 1;
  struct ncclSocketComm* comm = (struct ncclSocketComm*)sendComm;
  *request = NULL;
  NCCLCHECK(socketSend(comm->fd, &size, sizeof(int)));
  NCCLCHECK(socketSend(comm->fd, data, size));
  return 0;
}

int ncclSocketIrecv(void* recvComm, void* data, int size, int type, void** request) {
  if (type != NCCL_PTR_HOST) return 1;
  struct ncclSocketComm* comm = (struct ncclSocketComm*)recvComm;
  int recvSize;
  NCCLCHECK(socketReceive(comm->fd, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d\n", recvSize, size);
    return ncclInternalError;
  }
  NCCLCHECK(socketReceive(comm->fd, data, min(recvSize, size)));
  struct ncclSocketRequest* recvReq = ncclSocketGetRequest(&comm->reqs);
  recvReq->size = recvSize;
  *request = recvReq;
  return 0;
}

int ncclSocketTest(void* request, int* done, int* size) {
  *done = 1;
  struct ncclSocketRequest *r = (struct ncclSocketRequest*)request;
  if (r) {
    if (size) *size = r->size;
    r->used = 0;
  }
  return 0;
}

int ncclSocketClose(void* opaqueComm) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)opaqueComm;
  if (comm) {
    free(comm->reqs.requests);
    close(comm->fd);
    free(comm);
  }
  return 0;
}

ncclNet_t ncclNetSocket = {
  "Socket",
  ncclSocketDevices,
  ncclSocketPtrSupport,
  ncclSocketListen,
  ncclSocketConnect,
  ncclSocketAccept,
  ncclSocketIsend,
  ncclSocketIrecv,
  ncclSocketTest,
  ncclSocketClose,
  ncclSocketClose,
  ncclSocketClose
};
