#include "transport.h"
#if 0

typedef struct {
  char hostname[1024];
}socketInfo_t;

#include <stdint.h>

typedef struct {
  struct ip_addr ip_addr;
  uint16_t port;
}socketConnectInfo_t;


/* Fill infomation necessary to exchange between ranks to choose whether or not
 * to use this transport */
int socketFillInfo(ncclTinfo_t* opaqueInfo) {
  return 0;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
int socketSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  cudaMallocHost(&ring->mem, sizeof(struct ncclSendRecvMem+comm->buffSize));
  return 0;
}

/* Connect to this peer */
int socketConnectRecv(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
//  recv->head = ring->mem->head;
  return 0;
}
int socketConnectSend(struct ncclConnect* connectInfo, struct ncclConnector* send) {
//  send->buff = ring->mem->buff;
//  send->tail = ring->mem->tail;
  return 0;
}

struct ncclTransport socketTransport = {
  socketFillInfo,
  socketConnectSend,
  socketConnectRecv,
  socketSendProxy,
  socketRecvProxy
};

#endif

