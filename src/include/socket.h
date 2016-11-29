#ifndef COMMON_SOCKET_H_
#define COMMON_SOCKET_H_

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <errno.h>

#define SYSCHECK(call, name) do { \
  int ret = -1; \
  while (ret == -1) { \
    SYSCHECKVAL(call, name, ret); \
    if (ret == -1) { \
      INFO("Got retcode %d, retrying", errno); \
    }\
  } \
} while (0);

#define SYSCHECKVAL(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) { \
    WARN("call to " name " failed with ret %d", errno); \
    perror(name); \
    return ncclSystemError; \
  } \
} while (0);

typedef enum {
  GETIP_ENV = 1,
  GETIP_NO_LO = 2,
  GETIP_WITH_LO = 3
}getIpMode_t;

static int getIpMode(getIpMode_t mode, struct in_addr* addr) {
  /* Get IP address */
  char* if_env_name = NULL;
  if (mode == GETIP_ENV) {
    if_env_name = getenv("NCCL_SOCKET_IFNAME"); // Force a specific interface
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
    INFO("using interface %s, IP %s", interface->ifa_name, inet_ntoa(sa->sin_addr));
    break;
  }
  freeifaddrs(interfaces);
  if (!found)
    if (mode == GETIP_ENV) {
    // Env was set but we didn't find the interface ; fail.
    WARN("interface %s not found.", if_env_name);
    return -1;
  } else if (mode == GETIP_WITH_LO) {
    WARN("no usable interface found.");
  }
  return found;
}

static ncclResult_t getIpAddr(struct in_addr* addr) {
  int ret = getIpMode(GETIP_ENV, addr);
  if (ret == 0) { // No env
    ret = getIpMode(GETIP_NO_LO, addr)
      || getIpMode(GETIP_WITH_LO, addr);
  }
  return (ret == 1) ? ncclSuccess : ncclInternalError;
}

static ncclResult_t createListenSocket(int *fd, uint16_t *port) {
  /* Create socket and bind it to a port */
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == -1) {
    WARN("Socket creation failed");
    return ncclSystemError;
  }
  struct sockaddr_in sa_in = { AF_INET, INADDR_ANY, 0 /* Any port */ };
  SYSCHECK(bind(sockfd, (struct sockaddr*)&sa_in, sizeof(sa_in)), "bind");

  /* Get Port */
  socklen_t size = sizeof(struct sockaddr_in);
  SYSCHECK(getsockname(sockfd, (struct sockaddr*)&sa_in, &size), "getsockname");
  *port = sa_in.sin_port;
  INFO("Listening on port %d", *port);

  /* Put the socket in listen mode */
  SYSCHECK(listen(sockfd, MAXRANKS), "listen");
  *fd = sockfd;
  return ncclSuccess;
}

struct socketAddress {
  struct in_addr ip_addr;
  uint16_t port;
};

static ncclResult_t connectAddress(struct socketAddress* address, int* fd) {
  /* Connect to a hostname / port */
  *fd = socket(AF_INET, SOCK_STREAM, 0);
  if (*fd == -1) {
    WARN("Socket creation failed");
    return ncclSystemError;
  }
  const int one = 1;
  SYSCHECK(setsockopt(*fd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)), "setsockopt");
/*  const int bufsize = 128*1024;
  SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_SNDBUF, (char*)&bufsize, sizeof(int)), "setsockopt");
  SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_RCVBUF, (char*)&bufsize, sizeof(int)), "setsockopt");*/
  struct sockaddr_in remote;
  remote.sin_family = AF_INET;
  remote.sin_addr = address->ip_addr;
  remote.sin_port = address->port;
  SYSCHECK(connect(*fd, (struct sockaddr*)&remote, sizeof(remote)), "connect");
  return ncclSuccess;
}

static ncclResult_t socketReceive(int fd, void* ptr, int size) {
  char* data = (char*)ptr;
  int offset = 0;
  while (offset < size) {
    int recvsize;
    SYSCHECKVAL(recv(fd, data, size-offset, 0), "recv", recvsize);
    if (recvsize == 0) {
      WARN("Connection close by remote peer");
      return ncclSystemError;
    }
    if (recvsize == -1) {
      INFO("Recv : got retcode %d, retrying", errno);
      continue;
    }
    data += recvsize;
    offset += recvsize;
  }
  return ncclSuccess;
}

static ncclResult_t socketSend(int fd, void* ptr, int size) {
  char* data = (char*)ptr;
  int offset = 0;
  while (offset < size) {
    int sendsize;
    SYSCHECKVAL(write(fd, data, size-offset), "write", sendsize);
    if (sendsize == -1) {
      INFO("Send : got retcode %d, retrying", errno);
      continue;
    }
    data += sendsize;
    offset += sendsize;
  }
  return ncclSuccess;
}

#endif
