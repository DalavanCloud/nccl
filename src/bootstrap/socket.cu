#if 0
#include "nccl.h"
#include "core.h"
#include "net.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <errno.h>

struct netAddress {
  struct in_addr ip_addr;
  uint16_t port;
};

struct netEndpoint {
  struct netAddress address;
  int sendfd;
  int recvfd;
};

struct netModule {
  int nranks;
  int myrank;
  struct netEndpoint* endpoints;
};

static int netRootFd;

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

static ncclResult_t createEndpoint(struct netEndpoint* endpoint) {
  /* Create socket and bind it to a port */
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == -1) {
    WARN("netSocket : Socket creation failed");
    return ncclSystemError;
  }
  struct sockaddr_in sa_in = { AF_INET, INADDR_ANY, 0 /* Any port */ };
  SYSCHECK(bind(sockfd, (struct sockaddr*)&sa_in, sizeof(sa_in)), "bind");

  /* Get Port */
  socklen_t size = sizeof(struct sockaddr_in);
  SYSCHECK(getsockname(sockfd, (struct sockaddr*)&sa_in, &size), "getsockname");
  endpoint->address.port = sa_in.sin_port;
  INFO("netSocket : Listening on port %d", endpoint->address.port);
 
  /* Get IP address */
  char* if_name = getenv("NCCL_NET_SOCKET_IFNAME"); // Force a specific interface
  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface; interface = interface->ifa_next) {
    if (if_name && strlen(if_name) > 1 && strcmp(interface->ifa_name, if_name) != 0)
      continue;
    if (strncmp(interface->ifa_name, "lo", 2) == 0)
      continue;
    if (interface->ifa_addr == NULL || interface->ifa_addr->sa_family != AF_INET)
      continue;
    struct sockaddr_in* sa = (struct sockaddr_in*)(interface->ifa_addr);
    endpoint->address.ip_addr = sa->sin_addr;
    found = 1;
    INFO("netSocket : Using interface %s, IP %s", interface->ifa_name, inet_ntoa(sa->sin_addr));
    break;
  }
  freeifaddrs(interfaces);
  if (!found) {
    if (if_name && strlen(if_name) > 0) {
      WARN("netSocket : interface %s not found.", if_name);
    } else {
      WARN("netSocket : no interface found.");;
    }
    return ncclInvalidArgument;
  }

  /* Put the socket in listen mode */
  SYSCHECK(listen(sockfd, MAXRANKS), "listen");
  endpoint->recvfd = sockfd;
  return ncclSuccess;
}

static ncclResult_t connectAddress(struct netAddress* address, int* fd) {
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

ncclResult_t netSocketGetUniqueId(struct ncclInternalUniqueId* id) {
  struct netEndpoint endpoint;
  ncclResult_t ret = createEndpoint(&endpoint);
  if (ret != ncclSuccess)
    return ret;
  id->jobId = (uint64_t)(endpoint.address.port) << 32 + (endpoint.address.ip_addr.s_addr);
  memcpy(id->data, &endpoint.address, sizeof(endpoint.address));
  strncmp(id->type, "socket", sizeof(id->type));
  /* Store the fd for the allgather in netInit */
  netRootFd = endpoint.recvfd;
  return ncclSuccess;
}

ncclResult_t netSocketNetInit(ncclInternalUniqueId* id, int myrank, int nranks, void** net) {
  struct netAddress* rootAddress = (struct netAddress*)id->data;
  
  struct netEndpoint myEndpoint;
  ncclResult_t ret = createEndpoint(&myEndpoint);
  if (ret != ncclSuccess)
    return ret;

  struct netEndpoint* allEndpoints = (struct netEndpoint*) malloc(sizeof(struct netEndpoint)*nranks);
  
  /* Allgather(myEndpoints,allEndpoints) through the root */
  if (myrank == 0) {
    /* Set my address */
    memcpy(&(allEndpoints[myrank].address), &(myEndpoint.address), sizeof(struct netAddress));

    /* Receive addresses from all ranks */
    struct sockaddr_in sockaddrs[nranks-1];
    socklen_t socklen = sizeof(struct sockaddr_in);
    int sockfds[nranks];
    for (int c=0; c<nranks-1; c++) {
      SYSCHECKVAL(accept(netRootFd, (struct sockaddr*)sockaddrs+c, &socklen), "accept", sockfds[c]);
      int rank;
      /* Receive the rank first */
      SYSCHECK(recv(sockfds[c], &rank, sizeof(int), 0), "recv");
      /* Then store the address of that rank */
      SYSCHECK(recv(sockfds[c], &(allEndpoints[rank].address), sizeof(struct netAddress), 0), "recv");
      INFO("Root : received rank %d on ip %s, port %d", rank, inet_ntoa(allEndpoints[rank].address.ip_addr), allEndpoints[rank].address.port);
      allEndpoints[rank].sendfd = allEndpoints[rank].recvfd = -1;
    }
   
    /* Send all addresses to all ranks */
    for (int c=0; c<nranks-1; c++) {
      for (int rank=0; rank<nranks; rank++) {
        SYSCHECK(write(sockfds[c], &(allEndpoints[rank].address), sizeof(struct netAddress)), "write");
      }
      close(sockfds[c]);
    }

    close(netRootFd);
  } else {
    /* Connect to the root */
    int rootFd;
    ncclResult_t ret = connectAddress(rootAddress, &rootFd);
    if (ret != ncclSuccess)
      return ret;

    /* Send our rank first */
    SYSCHECK(write(rootFd, &myrank, sizeof(int)), "write");
    /* Then our address */
    SYSCHECK(write(rootFd, &(myEndpoint.address), sizeof(struct netAddress)), "write");

    /* Receive all addresses */
    for (int rank=0; rank<nranks; rank++) {
      SYSCHECK(recv(rootFd, &(allEndpoints[rank].address), sizeof(struct netAddress), 0), "recv");
      INFO("Rank %d : Received rank %d on host %s, port %d", myrank, rank, inet_ntoa(allEndpoints[rank].address.ip_addr), allEndpoints[rank].address.port);
      allEndpoints[rank].sendfd = allEndpoints[rank].recvfd = -1;
    }
    close(rootFd);
  }

  /* Set our local (listen) fd */
  allEndpoints[myrank].recvfd = myEndpoint.recvfd;

  /* Return the array of endpoints */
  struct netModule* netMod = (struct netModule*) malloc(sizeof(struct netModule));
  netMod->nranks = nranks;
  netMod->myrank = myrank;
  netMod->endpoints = allEndpoints;
  *net = (void*)netMod;
  return ncclSuccess;
}

static int getSendFd(struct netModule* mod, int rank) {
  int *fd = &mod->endpoints[rank].sendfd;
  if (*fd == -1) {
    connectAddress(&mod->endpoints[rank].address, fd);
  }
  return *fd;
}

static int getRecvFd(struct netModule* mod, int rank) {
  int *fd = &mod->endpoints[rank].recvfd;
  if (*fd == -1) {
    struct sockaddr* sa;
    socklen_t socklen = sizeof(struct sockaddr);
    SYSCHECKVAL(accept(mod->endpoints[mod->myrank].recvfd, sa, &socklen), "accept", *fd);
  }
  return *fd;
}

ncclResult_t netSocketConnect(void* net, int rank) {
  struct netModule *netMod = (struct netModule *)net;
  if (getSendFd(netMod, rank) == -1 || getRecvFd(netMod, rank) == -1)
    return ncclInternalError;
  else
    return ncclSuccess;
}

ncclResult_t netSocketSend(void* net, int rank, void* data, int size) {
  struct netModule *netMod = (struct netModule *)net;
  int fd = getSendFd(netMod, rank);
  printf("rank %d, fd = %d\n", rank, fd);
  SYSCHECK(write(fd, data, size), "write");
  return ncclSuccess;
}

ncclResult_t netSocketRecv(void* net, int rank, void* data, int size) {
  struct netModule *netMod = (struct netModule *)net;
  int fd = getRecvFd(netMod, rank);
  SYSCHECK(recv(fd, data, size, 0), "recv");
  return ncclSuccess;
}

#define NETCHECK(call) \
  do { \
    ncclResult_t retval = call; \
    if (retval != ncclSuccess) \
      return retval; \
  } while (0);

ncclResult_t netSocketAllGather(void* net, void* data, void* alldata, int size) {
  /* Ring allgather */
  struct netModule *netMod = (struct netModule *)net;
  int prev = ( netMod->myrank - 1 + netMod->nranks ) % netMod->nranks;
  int next = ( netMod->myrank + 1 ) % netMod->nranks;
  NETCHECK(netSocketSend(net, next, data, size));
  for (int rank = prev; rank != netMod->myrank; rank = (rank == 0) ? netMod->nranks-1 : rank-1) {
    NETCHECK(netSocketRecv(net, prev, (char*)alldata + size * rank, size));
    NETCHECK(netSocketSend(net, next, (char*)alldata + size * rank, size));
  }
  NETCHECK(netSocketRecv(net, prev, (char*)alldata + size * netMod->myrank, size));
  return ncclSuccess;
}

ncclResult_t netSocketAllReduce(void* net, int* val, int* res) {
  /* Central Allreduce : Reduce to 0 first */
  struct netModule *netMod = (struct netModule *)net;
  int rank = netMod->myrank;
  if (rank != netMod->nranks - 1) {
    NETCHECK(netSocketRecv(net, rank+1, res, sizeof(int)));
    *res = min(*val, *res);
  }
    
  if (rank != 0) {
    NETCHECK(netSocketSend(net, rank-1, res, sizeof(int)));
    /* Now 0 has the data. Broadcast. */
    NETCHECK(netSocketRecv(net, rank-1, res, sizeof(int)));
  }

  if (rank != netMod->nranks - 1)
    NETCHECK(netSocketSend(net, rank+1, res, sizeof(int)));
  return ncclSuccess;
}

ncclNet_t ncclNetSocket = {
  netSocketGetUniqueId,
  netSocketNetInit,
  netSocketAllGather,
  netSocketAllReduce,
  netSocketConnect,
  netSocketSend,
  netSocketRecv
};
#endif
