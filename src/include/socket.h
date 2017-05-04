/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_H_
#define NCCL_SOCKET_H_

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <netdb.h>
#include <ifaddrs.h>

/* Common socket address storage structure for IPv4/IPv6 */
union socketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

/* Format a string representation of a (struct sockaddr *) socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
static inline const char *socketToString(struct sockaddr *saddr, char *buf) {
  if (buf == NULL || saddr == NULL) return NULL;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) { buf[0]='\0'; return buf; }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  (void) getnameinfo(saddr, sizeof(union socketAddress), host, NI_MAXHOST, service, NI_MAXSERV, NI_NUMERICHOST|NI_NUMERICSERV);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static inline int envSocketFamily(void) {
  int family = -1; // Family selection is not forced, will use first one found
  char* env = getenv("NCCL_SOCKET_FAMILY");
  if (env == NULL)
    return family;
  
  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET;  // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6; // IPv6
  return family;
}

static int findInterfaces(const char* ifNamePrefix, char* names, union socketAddress *addrs, int maxIfNameSize, int maxIfs) {
  char line[1024];
  // Allow user to force the INET socket family selection
  int sock_family = envSocketFamily();
  bool searchNot = (strlen(ifNamePrefix) > 0 && ifNamePrefix[0] == '^');
  if (searchNot) /* Skip the '^' */ ifNamePrefix++;
  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;
    if (strncmp("lo", interface->ifa_name, strlen("lo")) == 0) continue; // Do not use loopback interfaces

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

//    INFO("Found interface %s:%s", interface->ifa_name, socketToString(interface->ifa_addr, line));

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family)
      continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    int matchLength = min((int)strlen(ifNamePrefix), maxIfNameSize);
    int match = strncmp(interface->ifa_name, ifNamePrefix, matchLength);
    if ((match == 0) ^ searchNot) {
      // Store the interface name
      strncpy(names+found*maxIfNameSize, interface->ifa_name, maxIfNameSize);
      // Store the IP address
      /* IPv4/IPv6 support */
      int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
      memcpy(addrs+found, interface->ifa_addr, salen);
      INFO("NET : Using interface %s:%s", interface->ifa_name, socketToString(interface->ifa_addr, line));
      found++;
      if (found == maxIfs) break;
    }
  }
  freeifaddrs(interfaces);
  return found;
}


static ncclResult_t createListenSocket(int *fd, union socketAddress *localAddr) {
  /* IPv4/IPv6 support */
  int family = localAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);

  /* Create socket and bind it to a port */
  int sockfd = socket(family, SOCK_STREAM, 0);
  if (sockfd == -1) {
    WARN("Socket creation failed");
    return ncclSystemError;
  }

  // localAddr port should be 0 (Any port)
  SYSCHECK(bind(sockfd, &localAddr->sa, salen), "bind");

  /* Get the assigned Port */
  socklen_t size = salen;
  SYSCHECK(getsockname(sockfd, &localAddr->sa, &size), "getsockname");

#if ENABLE_TRACE
  char line[1024];
  TRACE("Listening on socket %s", socketToString(&localAddr->sa, line));
#endif

  /* Put the socket in listen mode */
  SYSCHECK(listen(sockfd, 128), "listen");
  *fd = sockfd;
  return ncclSuccess;
}

static ncclResult_t connectAddress(union socketAddress* remoteAddr, union socketAddress* localAddr, int* fd) {
  /* IPv4/IPv6 support */
  int family = localAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);

  /* Connect to a hostname / port */
  *fd = socket(family, SOCK_STREAM, 0);
  if (*fd == -1) {
    WARN("Socket creation failed");
    return ncclSystemError;
  }

  // localAddr port should be 0 (Any port)
  SYSCHECK(bind(*fd, &localAddr->sa, salen), "bind");

  const int one = 1;
  SYSCHECK(setsockopt(*fd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)), "setsockopt");

/*  const int bufsize = 128*1024;
  SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_SNDBUF, (char*)&bufsize, sizeof(int)), "setsockopt");
  SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_RCVBUF, (char*)&bufsize, sizeof(int)), "setsockopt");*/

#if ENABLE_TRACE
  char line[1024];
  TRACE("Connecting to socket %s", socketToString(&remoteAddr->sa, line));
#endif

  SYSCHECK(connect(*fd, &remoteAddr->sa, salen), "connect");
  return ncclSuccess;
}

static ncclResult_t socketReceive(int fd, void* ptr, int size) {
  char* data = (char*)ptr;
  int offset = 0;
  while (offset < size) {
    int recvsize;
    SYSCHECKVAL(recv(fd, data, size-offset, 0), "recv", recvsize);
    if (recvsize == 0) {
      WARN("Connection closed by remote peer");
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
