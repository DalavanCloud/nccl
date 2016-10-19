#include "core.h"
#include "transport.h"
#include <cuda_runtime.h>

struct p2pInfo {
  int cudaDev;
  int pid;
  uint64_t hostHash;
  int hostNumber;
};

struct p2pConnectInfo {
  int direct;
  union {
    struct ncclSendRecvMem* directPtr;
    cudaIpcMemHandle_t devIpc;
  };
};

#include <sys/types.h>
#include <unistd.h>

/* Fill infomation necessary to exchange between ranks to choose whether or not
 * to use this transport */
int p2pFillInfo(ncclTinfo_t* opaqueInfo) {
  struct p2pInfo* info = (struct p2pInfo*)opaqueInfo;
  static_assert(sizeof(struct p2pInfo) <= sizeof(ncclTinfo_t), "p2p Info too large");
  cudaGetDevice(&info->cudaDev);
  //info->pid = getpid();
  char hostname[1024];
  gethostname(hostname, 1024);
  info->hostHash=hostHash(hostname);
  info->hostNumber=hostNumber(hostname);
  return 0;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
int p2pSetupSend(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct p2pInfo* myInfo = (struct p2pInfo*)myOpaqueInfo;
  struct p2pInfo* peerInfo = (struct p2pInfo*)peerOpaqueInfo;
  if (myInfo->hostHash == peerInfo->hostHash)
    return 1;

  int p2p = myInfo->cudaDev == peerInfo->cudaDev ? 1 : 0;
  if (p2p == 0) {
    if (cudaDeviceCanAccessPeer(&p2p, myInfo->cudaDev, peerInfo->cudaDev) != cudaSuccess) {
      INFO("peer query failed between dev %d and dev %d",
        myInfo->cudaDev, peerInfo->cudaDev);
      p2p = 0;
    }
  }
  if (p2p == 0)
    return 1;

  struct p2pConnectInfo info;
  if (myInfo->pid == peerInfo->pid) {
    info.direct = 1;
    info.directPtr = ring->sendrecv.devMem;
  } else {
    // Use CUDA IPC
    info.direct = 0;
    if (cudaIpcGetMemHandle(&info.devIpc, (void*)ring->sendrecv.devMem) != cudaSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d", ring->rank, peerInfo->cudaDev);
      return 1;
    }
  }
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return 0;
}

int p2pSetupRecv(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t*peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct p2pInfo* myInfo = (struct p2pInfo*)myOpaqueInfo;
  struct p2pInfo* peerInfo = (struct p2pInfo*)peerOpaqueInfo;
  if (myInfo->hostHash == peerInfo->hostHash)
    return 1;

  int p2p = myInfo->cudaDev == peerInfo->cudaDev ? 1 : 0;
  if (p2p == 0) {
    if (cudaDeviceCanAccessPeer(&p2p, myInfo->cudaDev, peerInfo->cudaDev) != cudaSuccess) {
      INFO("peer query failed between dev %d and dev %d",
        myInfo->cudaDev, peerInfo->cudaDev);
      p2p = 0;
    }
  }
  if (p2p == 0)
    return 1;

  struct p2pConnectInfo info;
  if (myInfo->pid == peerInfo->pid) {
    info.direct = 1;
    info.directPtr = ring->sendrecv.devMem;
  } else {
    // Use CUDA IPC
    info.direct = 0;
    if (cudaIpcGetMemHandle(&info.devIpc, (void*)ring->sendrecv.devMem) != cudaSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d", ring->rank, peerInfo->cudaDev);
      return 1;
    }
  }
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return 0;
}

/* Connect to this peer */
int p2pConnectSend(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  struct ncclSendRecvMem* remDevMem;
  if (info->direct) {
    remDevMem = info->directPtr;
    send->conn.direct = 1;
    send->conn.ptrExchange = &(remDevMem->ptrExchange);
  } else {
    cudaError_t err = cudaIpcOpenMemHandle((void**)&remDevMem,
          info->devIpc, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
      WARN("failed to open CUDA IPC handle : %s", cudaGetErrorString(err));
      return 1;
    }
  }
  send->conn.buff = remDevMem->buff;
  send->conn.tail = &remDevMem->tail;
  // send->conn->head should have been set to devMem already
  return 0;
}

int p2pConnectRecv(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  struct ncclSendRecvMem* remDevMem;
  if (info->direct) {
    remDevMem = info->directPtr;
    recv->conn.direct = 1;
    recv->conn.ptrExchange = &remDevMem->ptrExchange;
  } else {
    cudaError_t err = cudaIpcOpenMemHandle((void**)&remDevMem,
          info->devIpc, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
      WARN("failed to open CUDA IPC handle : %s",
          cudaGetErrorString(err));
      return 1;
    }
  }
  // recv->conn->buff should have been set to devMem already
  // recv->conn->tail should have been set to devMem already
  recv->conn.head = &remDevMem->head;
  return 0;
}

struct ncclTransport p2pTransport = {
  p2pFillInfo,
  { p2pSetupSend, p2pConnectSend, NULL },
  { p2pSetupRecv, p2pConnectRecv, NULL }
};


