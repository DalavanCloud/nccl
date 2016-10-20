/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "bootstrap.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <errno.h>

struct socketAddress {
  struct in_addr ip_addr;
  uint16_t port;
};

struct socketId {
  struct socketAddress addr;
  pid_t pid;
  int* lock;
  int fd;
};

#define SYSCHECK(call, name) do { \
  int ret; \
  SYSCHECKVAL(call, name, ret); \
} while (0);

#define SYSCHECKVAL(call, name, retval) do { \
  retval = call; \
  if (retval == -1) { \
    WARN("netSocket : call to " name " failed with ret %d", errno); \
    perror(name); \
    return ncclSystemError; \
  } \
} while (0);

static ncclResult_t createListenSocket(int *fd, uint16_t *port) {
  /* Create socket and bind it to a port */
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == -1) {
    WARN("bootstrapSocket : Socket creation failed");
    return ncclSystemError;
  }
  struct sockaddr_in sa_in = { AF_INET, INADDR_ANY, 0 /* Any port */ };
  SYSCHECK(bind(sockfd, (struct sockaddr*)&sa_in, sizeof(sa_in)), "bind");

  /* Get Port */
  socklen_t size = sizeof(struct sockaddr_in);
  SYSCHECK(getsockname(sockfd, (struct sockaddr*)&sa_in, &size), "getsockname");
  *port = sa_in.sin_port;
  INFO("bootstrapSocket : Listening on port %d", *port);

  /* Put the socket in listen mode */
  SYSCHECK(listen(sockfd, MAXRANKS), "listen");
  *fd = sockfd;
  return ncclSuccess;
}

typedef enum {
  GETIP_ENV = 1,
  GETIP_NO_LO = 2,
  GETIP_WITH_LO = 3
}getIpMode_t;

static int getIpMode(getIpMode_t mode, struct in_addr* addr) {
  /* Get IP address */
  char* if_env_name = NULL;
  if (mode == GETIP_ENV) {
    if_env_name = getenv("NCCL_NET_SOCKET_IFNAME"); // Force a specific interface
    if (if_env_name == NULL || strlen(if_env_name) == 0)
      // Env not set, skip this call
      return 0;
  }

  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface; interface = interface->ifa_next) {
    if (mode == GETIP_ENV && strcmp(interface->ifa_name, if_env_name) != 0)
      continue;
    if (mode == GETIP_NO_LO && strncmp(interface->ifa_name, "lo", 2) == 0)
      continue;
    if (interface->ifa_addr == NULL || interface->ifa_addr->sa_family != AF_INET)
      continue;
    struct sockaddr_in* sa = (struct sockaddr_in*)(interface->ifa_addr);
    *addr = sa->sin_addr;
    found = 1;
    INFO("bootstrapSocket : Using interface %s, IP %s", interface->ifa_name, inet_ntoa(sa->sin_addr));
    break;
  }
  freeifaddrs(interfaces);
  if (!found)
    if (mode == GETIP_ENV) {
    // Env was set but we didn't find the interface ; fail.
    WARN("bootstrapSocket : interface %s not found.", if_env_name);
    return -1;
  } else if (mode == GETIP_WITH_LO) {
    WARN("bootstrapSocket : no usable interface found.");
  }
  return found;
}

static ncclResult_t getIpAddr(struct in_addr* addr) {
  int ret = getIpMode(GETIP_ENV, addr);
  if (ret == 1) { // No env
    ret = getIpMode(GETIP_NO_LO, addr)
      || getIpMode(GETIP_WITH_LO, addr);
  }
  return (ret == 1) ? ncclSuccess : ncclInternalError;
}

ncclResult_t bootstrapSocketGetUniqueId(ncclUniqueId* out) {
  socketId* id = (socketId*)out;
  NCCLCHECK(createListenSocket(&id->fd, &id->addr.port));
  NCCLCHECK(getIpAddr(&(id->addr.ip_addr)));
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

static ncclResult_t connectAddress(struct socketAddress* address, int* fd) {
  /* Connect to a hostname / port */
  *fd = socket(AF_INET, SOCK_STREAM, 0);
  if (*fd == -1) {
    WARN("Socket creation failed");
    return ncclSystemError;
  }
  struct sockaddr_in remote;
  remote.sin_family = AF_INET;
  remote.sin_addr = address->ip_addr;
  remote.sin_port = address->port;
  SYSCHECK(connect(*fd, (struct sockaddr*)&remote, sizeof(remote)), "connect");
  return ncclSuccess;
}


ncclResult_t bootstrapSocketInit(ncclUniqueId* commId, int rank, int nranks, void** commState) {
  struct socketId* id = (struct socketId*)commId;
  int root = 0;
  int fds[nranks];

  pid_t pid = getpid();
  if (pid == id->pid) {
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
      SYSCHECK(recv(sockfd, &rank, sizeof(int), 0), "recv");
      /* Then store the fd of that rank */
      fds[rank] = sockfd;
    }
    close(id->fd);
  } else {
    /* Connect to the root */
    NCCLCHECK(connectAddress(&id->addr, &fds[0]));

    /* Send our rank */
    SYSCHECK(write(fds[0], &rank, sizeof(int)), "write");
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
      SYSCHECK(recv(state->fds[r], data+r*size, size, 0), "recv");
    }
    for (int r=1; r<state->nranks; r++) {
      SYSCHECK(write(state->fds[r], data, size*state->nranks), "write");
    }
  } else {
    SYSCHECK(write(state->fds[0], data+state->rank*size, size), "write");
    SYSCHECK(recv(state->fds[0], data, size*state->nranks, 0), "recv");
  }
  return ncclSuccess;
}

ncclResult_t bootstrapSocketRingExchange(void* commState, void* prevNextData, int size) {
  struct socketState* state = (struct socketState*)commState;
  char* mydata = (char*)prevNextData;
  if (state->root) {
    char* data = (char*)malloc(size*2*state->nranks);
    // Copy root prev/next 
    memcpy(data+(state->nranks-1)*2*size+size, mydata, size);
    memcpy(data+2*size, mydata+size, size);

    // Receive from others directly at the right place
    for (int r=1; r<state->nranks; r++) {
      SYSCHECK(recv(state->fds[r], data+(r-1)*2*size+size, size, 0), "recv");
      SYSCHECK(recv(state->fds[r], data+((r+1)%state->nranks)*2*size, size, 0), "recv");
    }

    // Get root prev/next
    memcpy(mydata, data, 2*size);

    // Send to all
    for (int r=1; r<state->nranks; r++) {
      SYSCHECK(write(state->fds[r], data+r*2*size, 2*size), "write");
    }

    free(data);
  } else {
    SYSCHECK(write(state->fds[0], mydata, 2*size), "write");
    SYSCHECK(recv(state->fds[0], mydata, 2*size, 0), "recv");
  }
  return ncclSuccess;
}

struct ncclBootstrap bootstrapSocket = {
  bootstrapSocketGetUniqueId,
  bootstrapSocketInit,
  bootstrapSocketAllGather,
  bootstrapSocketRingExchange
};
