/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "socket.h"

struct socketId {
  struct socketAddress addr;
  uint64_t hostHash;
  pid_t pid;
  int* lock;
  int fd;
};

ncclResult_t bootstrapSocketGetUniqueId(ncclUniqueId* out) {
  static_assert(sizeof(socketId) < sizeof(ncclUniqueId), "SocketId does not fit inside ncclUniqueId");
  socketId* id = (socketId*)out;
  NCCLCHECK(createListenSocket(&id->fd, &id->addr.port));
  NCCLCHECK(getIpAddr(&(id->addr.ip_addr), NULL));
  char hostname[1024];
  getHostName(hostname, 1024);
  id->hostHash = getHostHash(hostname);
  id->pid = getpid();
  id->lock = (int*)malloc(sizeof(int));
  *(id->lock) = 0;
  return ncclSuccess;
}

struct socketState {
  int* fds;
  int root;
  int rank;
  int nranks;
};

ncclResult_t bootstrapSocketInit(ncclUniqueId* commId, int rank, int nranks, void** commState) {
  struct socketId* id = (struct socketId*)commId;
  int root = 0;
  int fds[nranks];

  char hostname[1024];
  getHostName(hostname, 1024);
  uint64_t hostHash = getHostHash(hostname);
  pid_t pid = getpid();
  if (hostHash == id->hostHash && pid == id->pid) {
    // Try to become root
    if (__sync_bool_compare_and_swap(id->lock, 0, 1))
      root = 1;
  }

  if (root) {
    /* Receive addresses from all ranks */
    struct sockaddr_in sockaddrs[nranks-1];
    socklen_t socklen = sizeof(struct sockaddr_in);
    for (int c=0; c<nranks-1; c++) {
      int sockfd;
      SYSCHECKVAL(accept(id->fd, (struct sockaddr*)sockaddrs+c, &socklen), "accept", sockfd);
      int rank;
      /* Receive the rank */
      NCCLCHECK(socketReceive(sockfd, &rank, sizeof(int)));
      /* Then store the fd of that rank */
      fds[rank] = sockfd;
    }
    close(id->fd);
    free(id->lock);
  } else {
    /* Connect to the root */
    NCCLCHECK(connectAddress(&id->addr, &fds[0]));

    /* Send our rank */
    NCCLCHECK(socketSend(fds[0], &rank, sizeof(int)));
  }
  
  struct socketState* state = (struct socketState*)malloc(sizeof(struct socketState));
  state->root = root;
  if (root) {
    state->fds = (int*)malloc(sizeof(int)*nranks);
    for (int r=0; r<nranks; r++)
      state->fds[r] = fds[r];
  } else {
    state->fds = (int*)malloc(sizeof(int));
    state->fds[0] = fds[0];
  }
  state->rank = rank;
  state->nranks = nranks;
  *commState = state;
  return ncclSuccess;
}

ncclResult_t bootstrapSocketAllGather(void* commState, void* allData, int size) {
  struct socketState* state = (struct socketState*)commState;
  char* data = (char*)allData;
  if (state->root) {
    for (int r=1; r<state->nranks; r++) {
      NCCLCHECK(socketReceive(state->fds[r], data+r*size, size));
    }
    for (int r=1; r<state->nranks; r++) {
      NCCLCHECK(socketSend(state->fds[r], data, size*state->nranks));
    }
  } else {
    NCCLCHECK(socketSend(state->fds[0], data+state->rank*size, size));
    NCCLCHECK(socketReceive(state->fds[0], data, size*state->nranks));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapSocketRingExchange(void* commState, void* prevNextData, int prev, int next, int size) {
  struct socketState* state = (struct socketState*)commState;
  char* mydata = (char*)prevNextData;
  int prev_offset = prev*2*size+size, next_offset = next*2*size;
  if (state->root) {
    char* data = (char*)malloc(size*2*state->nranks);
    // Copy root prev/next 
    memcpy(data, mydata, 2*size);

    // Receive from all and build total table
    for (int r=1; r<state->nranks; r++) {
      NCCLCHECK(socketReceive(state->fds[r], data+r*2*size, 2*size));
    }

    // Get root prev/next
    memcpy(mydata, data+prev_offset, size);
    memcpy(mydata+size, data+next_offset, size);

    // Get prev/next request from everyone and answer.
    for (int r=1; r<state->nranks; r++) {
      int offset;
      NCCLCHECK(socketReceive(state->fds[r], &offset, sizeof(int)));
      NCCLCHECK(socketSend(state->fds[r], data+offset, size));
      NCCLCHECK(socketReceive(state->fds[r], &offset, sizeof(int)));
      NCCLCHECK(socketSend(state->fds[r], data+offset, size));
    }

    free(data);
  } else {
    // Send data to root
    NCCLCHECK(socketSend(state->fds[0], mydata, 2*size));
    // Receive prev and next data
    NCCLCHECK(socketSend(state->fds[0], &prev_offset, sizeof(int)));
    NCCLCHECK(socketReceive(state->fds[0], mydata, size));
    NCCLCHECK(socketSend(state->fds[0], &next_offset, sizeof(int)));
    NCCLCHECK(socketReceive(state->fds[0], mydata+size, size));
  }
  return ncclSuccess;
}

struct ncclBootstrap bootstrapSocket = {
  bootstrapSocketGetUniqueId,
  bootstrapSocketInit,
  bootstrapSocketAllGather,
  bootstrapSocketRingExchange
};
