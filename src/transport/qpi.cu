#include "transport.h"

#if 0

struct qpiInfo {
  int cudaDev;
  int pid;
  char hostname[1024];
};

struct qpiConnectInfo {
  int id;
  int rank;
};

struct qpiResources {
  struct ncclSendRecvMem* hostMem;
};

/* Fill infomation necessary to exchange between ranks to choose whether or not
 * to use this transport */
int qpiFillInfo(ncclTinfo_t* opaqueInfo) {
  //qpiInfo_t* info = (qpiInfo_t*)opaqueInfo;
  assert(sizeof(qpiInfo_t) <= sizeof(ncclTinfo_t));
  //info->pid = getpid();
  //gethostname(info->hostname);
  return 0;
}

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
int qpiSetupSend(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct qpiInfo* myInfo = (struct qpiInfo*)myOpaqueInfo;
  struct qpiInfo* peerInfo = (struct qpiInfo*)peerOpaqueInfo;
  if (strncmp(myInfo->hostname, peerInfo->hostname, 1024) != 0)
    return 1;

  char filename[1024];
  sprintf(filename, "/dev/shm/nccl-%d-%d", ring->id, ring->rank);
  int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
  if (fd == -1) {
    WARN("Unable to create shared memory %s\n", filename);
    return 1;
  }
  struct qpiResources* resources = (struct qpiResources*) malloc(sizeof(qpiResources_t));
  ring->sendrecv.recv.transportResources = resources;
  resources->fd = fd;
  resources->hostMem = mmap(0, FILESIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (resources->hostMem == MAP_FAILED) {
    WARN("Unable to map shared memory");
    close(fd);
    return 1;
  }

  struct qpiConnectInfo info;
  info.id = ring->id; info.rank = ring->rank;
  memcpy(connectInfo, &info, sizeof(qpiConnectInfo_t));
  return 0;
}

int qpiSetupRecv(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct qpiInfo* myInfo = (struct qpiInfo*)myOpaqueInfo;
  struct qpiInfo* peerInfo = (struct qpiInfo*)peerOpaqueInfo;
  if (strncmp(myInfo->hostname, peerInfo->hostname, 1024) != 0)
    return 1;

  char filename[1024];
  sprintf(filename, "/dev/shm/nccl-%d-%d", ring->id, ring->rank);
  int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
  if (fd == -1) {
    WARN("Unable to create shared memory %s\n", filename);
    return 1;
  }
  struct qpiResources* resources = (struct qpiResources*) malloc(sizeof(qpiResources_t));
  ring->sendrecv.recv.transportResources = resources;
  resources->fd = fd;
  resources->hostMem = mmap(0, FILESIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (resources->hostMem == MAP_FAILED) {
    WARN("Unable to map shared memory");
    close(fd);
    return 1;
  }

  struct qpiConnectInfo info;
  info.id = ring->id; info.rank = ring->rank;
  memcpy(connectInfo, &info, sizeof(qpiConnectInfo_t));
  return 0;
}

/* Connect to this peer */
int qpiConnectRecv(struct ncclConnect* connectInfo, struct ncclConnector_t connector) {
  qpiConnectInfo_t* info = (qpiConnectInfo_t*)connectInfo;
  connector->head = &info->hostMem->head;
  return 0;
}

int qpiConnectSend(struct ncclConnect* connectInfo, struct ncclConnector_t connector) {
  qpiConnectInfo_t* info = (qpiConnectInfo_t*)connectInfo;
  char filename[1024];
  sprintf(filename, "/dev/shm/nccl-%d-%d", info->id, info->rank);
  int fd = open(filename);
  struct ncclSendRecvMem remHostMem = mmap(fd);
  connector->buff = remHostMem->buff;
  connector->tail = &remHostMem->tail;
  return 0;
}

struct ncclTransport qpiTransport = {
  qpiFillInfo,
  qpiConnect,
  NULL, // No Send proxy
  qpiRecvProxy
};
#endif
