#include "core.h"
#include "utils.h"
#include "transport.h"
#include <unistd.h>
#include <cuda_runtime.h>

struct p2pInfo {
  int rank;
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

/* Fill infomation necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t p2pFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  struct p2pInfo* info = (struct p2pInfo*)opaqueInfo;
  static_assert(sizeof(struct p2pInfo) <= sizeof(ncclTinfo_t), "p2p Info too large");
  info->rank = rank;
  CUDACHECK(cudaGetDevice(&info->cudaDev));
  info->pid = getpid();
  char hostname[1024];
  getHostName(hostname, 1024);
  info->hostHash=getHostHash(hostname);
  info->hostNumber=getHostNumber(hostname);
  return ncclSuccess;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t p2pSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring, int* select) {
  struct p2pInfo* myInfo = (struct p2pInfo*)myOpaqueInfo;
  struct p2pInfo* peerInfo = (struct p2pInfo*)peerOpaqueInfo;
  if (myInfo->hostHash != peerInfo->hostHash) {
    *select = 0;
    return ncclSuccess;
  }

  int p2p = myInfo->cudaDev == peerInfo->cudaDev ? 1 : 0;
  if (p2p == 0) {
    if (cudaDeviceCanAccessPeer(&p2p, myInfo->cudaDev, peerInfo->cudaDev) != cudaSuccess) {
      INFO("peer query failed between dev %d and dev %d",
        myInfo->cudaDev, peerInfo->cudaDev);
      p2p = 0;
    }
  }
  if (p2p == 0) {
    *select = 0;
    return ncclSuccess;
  }

  struct p2pConnectInfo info;
  if (myInfo->pid == peerInfo->pid) {
    info.direct = 1;
    info.directPtr = ring->devMem;
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      INFO("%d [%d] -> %d [%d] via P2P/common device", myInfo->rank, myInfo->cudaDev, peerInfo->rank, peerInfo->cudaDev);
    } else {
      // Enable P2P access
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d: %s",
            peerInfo->cudaDev, cudaGetErrorString(err));
        // We could return an error, but maybe it is better to gracefully disable
        // p2p and fall back on something else.
        *select = 0;
        return ncclSuccess;
      }
      INFO("%d [%d] -> %d [%d] via P2P/direct pointer", myInfo->rank, myInfo->cudaDev, peerInfo->rank, peerInfo->cudaDev);
    }
  } else {
    info.direct = 0;
    // Map IPC and enable P2P access
    if (cudaIpcGetMemHandle(&info.devIpc, (void*)ring->devMem) != cudaSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d", ring->rank, peerInfo->cudaDev);
      // We could return an error, but maybe it is better to gracefully disable
      // p2p and fall back on something else.
      *select = 0;
      return ncclSuccess;
    }
    INFO("%d [%d] -> %d [%d] via P2P/IPC", myInfo->rank, myInfo->cudaDev, peerInfo->rank, peerInfo->cudaDev);
  }
  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  *select = 1;
  return ncclSuccess;
}

static ncclResult_t p2pConnect(struct ncclConnect* connectInfo, struct ncclConnector* connector, struct ncclSendRecvMem** remDevMem) {
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  if (info->direct) {
    *remDevMem = info->directPtr;
    connector->conn.direct = 1;
    connector->conn.ptrExchange = &((*remDevMem)->ptrExchange);
  } else {
    cudaError_t err = cudaIpcOpenMemHandle((void**)remDevMem,
          info->devIpc, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
      WARN("failed to open CUDA IPC handle : %s",
          cudaGetErrorString(err));
      return ncclUnhandledCudaError;
    }
  }
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t p2pConnectSend(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  struct ncclSendRecvMem* remDevMem;
  NCCLCHECK(p2pConnect(connectInfo, send, &remDevMem));
  send->conn.buff = remDevMem->buff;
  send->conn.tail = &remDevMem->tail;
  send->conn.opCount = &remDevMem->opCount;
  // send->conn->head should have been set to devMem already
  return ncclSuccess;
}

ncclResult_t p2pConnectRecv(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  struct ncclSendRecvMem* remDevMem;
  NCCLCHECK(p2pConnect(connectInfo, recv, &remDevMem));
  // recv->conn->buff should have been set to devMem already
  // recv->conn->tail should have been set to devMem already
  // recv->conn->opCount should have been set to devMem already
  recv->conn.head = &remDevMem->head;
  return ncclSuccess;
}

struct ncclTransport p2pTransport = {
  p2pFillInfo,
  { p2pSetup, p2pConnectSend, NULL },
  { p2pSetup, p2pConnectRecv, NULL }
};


