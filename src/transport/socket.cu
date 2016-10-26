#include "core.h"
#include "transport.h"
#include <cuda_runtime.h>
#include "socket.h"
#include <assert.h>

struct socketInfo {
  int rank;
  int listen_fd;
  struct socketAddress connect_addr;
};

struct socketResourcesSend {
  int fd;
  cudaStream_t stream;
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
};

#define MAXSTEPS 8

struct socketResourcesRecv {
  int listen_fd;
  int fd;
  cudaStream_t stream;
  cudaEvent_t syncEvent[MAXSTEPS];
  struct ncclSendRecvMem* hostMem;
  struct ncclSendRecvMem* devHostMem;
};

/* Fill infomation necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t socketFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  struct socketInfo* info = (struct socketInfo*)opaqueInfo;
  static_assert(sizeof(struct socketInfo) <= sizeof(ncclTinfo_t), "socket Info too large");
  info->rank = rank;
  NCCLCHECK(createListenSocket(&info->listen_fd, &info->connect_addr.port));
  NCCLCHECK(getIpAddr(&(info->connect_addr.ip_addr)));
  return ncclSuccess;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t socketSetupSend(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring, int* select) {
  struct socketResourcesSend* resources = (struct socketResourcesSend*) malloc(sizeof(struct socketResourcesSend));
  ring->send.transportResources = resources;

  // Create stream for proxy
  CUDACHECK(cudaStreamCreateWithFlags(&resources->stream, cudaStreamNonBlocking));

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  CUDACHECK(cudaHostAlloc(&resources->hostMem, size, cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));

  struct socketInfo* myInfo = (struct socketInfo*)myOpaqueInfo;
  struct socketInfo* peerInfo = (struct socketInfo*)peerOpaqueInfo;
  INFO("%d -> %d via socket", myInfo->rank, peerInfo->rank);
  // Just pass the socket info through
  static_assert(sizeof(struct socketInfo) <= sizeof(struct ncclConnect), "socket Connect Info is too big");
  memcpy(connectInfo, myOpaqueInfo, sizeof(struct socketInfo));
  *select = 1;
  return ncclSuccess;
}

ncclResult_t socketSetupRecv(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring, int* select) {
  struct socketResourcesRecv* resources = (struct socketResourcesRecv*) malloc(sizeof(struct socketResourcesRecv));
  ring->recv.transportResources = resources;
  struct socketInfo* myInfo = (struct socketInfo*)myOpaqueInfo;
  resources->listen_fd = myInfo->listen_fd; 

  // Create stream for proxy
  CUDACHECK(cudaStreamCreateWithFlags(&resources->stream, cudaStreamNonBlocking));
  // And event
  for (int i=0; i<MAXSTEPS; i++)
    CUDACHECK(cudaEventCreate(resources->syncEvent+i));

  int size = offsetof(struct ncclSendRecvMem, buff)+ring->buffSize;
  CUDACHECK(cudaHostAlloc(&resources->hostMem, size, cudaHostAllocMapped));
  CUDACHECK(cudaHostGetDevicePointer(&resources->devHostMem, resources->hostMem, 0));
  
  // Just pass the socket info through
  memcpy(connectInfo, myOpaqueInfo, sizeof(struct socketInfo));
  *select = 1;
  return ncclSuccess;
}

ncclResult_t socketConnectSend(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct socketResourcesSend* resources = (struct socketResourcesSend*)send->transportResources;
  send->conn.buff = resources->devHostMem->buff;
  send->conn.tail = &resources->devHostMem->tail;

  // Setup receive proxy socket/pointers
  struct socketInfo* info = (struct socketInfo*)connectInfo;
  NCCLCHECK(connectAddress(&info->connect_addr, &resources->fd));
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t socketConnectRecv(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct socketResourcesRecv* resources = (struct socketResourcesRecv*)recv->transportResources;
  recv->conn.head = &resources->devHostMem->head;

  // We will finish the socket setup at beginning of Recv proxy
  resources->fd = 0;
  return ncclSuccess;
}

ncclResult_t socketSendProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct socketResourcesSend* resources = (struct socketResourcesSend*) (ring->send.transportResources);
  struct ncclSendRecvMem* devMem = ring->devMem;
  volatile int* prevTail = &resources->hostMem->tail;
  int* prevHead = &devMem->head;
  char* localBuff = resources->hostMem->buff;
  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  printf("Send proxy starting ...\n");
  int val = 1;
  while (val != 0) {
    printf("Waiting for prev to be ready ...\n");
    CUDACHECK(cudaMemcpyAsync(&val, prevHead, sizeof(int), cudaMemcpyDeviceToHost, resources->stream));
    CUDACHECK(cudaStreamSynchronize(resources->stream));
  }
  int head = 0;
  int offset = 0;

  printf("Send proxy pushing ...\n");
  while (head < args->nsteps) {
    while ((head - *prevTail) == 0);
    head++;
    printf("Sending data, head is %d\n", head);
    NCCLCHECK(socketSend(resources->fd, localBuff+offset, sliceSize));
    printf("Done\n");

    CUDACHECK(cudaMemcpyAsync(prevHead, &head, sizeof(int), cudaMemcpyHostToDevice, resources->stream));

    offset += sliceSize;
    if (offset == buffSize)
      offset = 0;
  }
  // Ensure all updates are pushed
  CUDACHECK(cudaStreamSynchronize(resources->stream));

  // Wait for last ack and reset
//  printf("Flags at end : %d | %d\n", head, *prevTail);
  *prevTail = 0;
  printf("Sendproxy exiting\n");
  return ncclSuccess;
}

ncclResult_t socketRecvProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct socketResourcesRecv* resources = (struct socketResourcesRecv*) (ring->recv.transportResources);
  struct ncclSendRecvMem* devMem = ring->devMem;
  int* nextTail = &devMem->tail;
  volatile int* nextHead = &resources->hostMem->head;
  char* localBuff = resources->hostMem->buff;
  char* nextBuff = devMem->buff;
  int buffSize = ring->buffSize;
  int sliceSize = buffSize / args->substeps;
  assert(MAXSTEPS >= args->substeps);

  if (resources->fd == 0) {
    struct sockaddr_in sockaddr;
    socklen_t socklen = sizeof(struct sockaddr_in);
    SYSCHECKVAL(accept(resources->listen_fd, (struct sockaddr*)&sockaddr, &socklen), "accept", resources->fd);
  }

  int val = 1;
  while (val != 0) {
    CUDACHECK(cudaMemcpyAsync(&val, nextTail, sizeof(int), cudaMemcpyDeviceToHost, resources->stream));
    CUDACHECK(cudaStreamSynchronize(resources->stream));
  }
  int head = 0;
  int offset = 0;

  while (head < args->nsteps) {
    while ((head - *nextHead) >= args->substeps);
    head++;
    CUDACHECK(cudaEventSynchronize(resources->syncEvent[head%args->substeps]));
    NCCLCHECK(socketReceive(resources->fd, localBuff+offset, sliceSize));
    CUDACHECK(cudaMemcpyAsync(nextBuff+offset, localBuff+offset, sliceSize, cudaMemcpyHostToDevice, resources->stream));
    CUDACHECK(cudaMemcpyAsync(nextTail, &head, sizeof(int), cudaMemcpyHostToDevice, resources->stream));
    CUDACHECK(cudaEventRecord(resources->syncEvent[head%args->substeps], resources->stream));

    offset += sliceSize;
    if (offset == buffSize)
      offset = 0;
  }
  // Ensure all updates are pushed
  CUDACHECK(cudaStreamSynchronize(resources->stream));

  // Wait for last ack and reset
//  printf("Flags at end : %d | %d %d\n", head, *nextHead, *prevTail);
  while (*nextHead < head);
  *nextHead = 0;

  return ncclSuccess;
}

struct ncclTransport socketTransport = {
  socketFillInfo,
  { socketSetupSend, socketConnectSend, socketSendProxy },
  { socketSetupRecv, socketConnectRecv, socketRecvProxy }
};
