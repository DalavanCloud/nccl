/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "utils.h"
#include "transport.h"
#include <unistd.h>
#include <cuda_runtime.h>
#include "nvmlwrap.h"
#include <ctype.h>

#define MAXNVLINKS 8

struct p2pInfo {
  int rank;
  int cudaDev;
  int pid;
  uint64_t hostHash;
  int hostNumber;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
};

struct p2pConnectInfo {
  int direct;
  union {
    struct ncclSendRecvMem* directPtr;
    cudaIpcMemHandle_t devIpc;
  };
};

#include <sys/types.h>

/* Fill information necessary to exchange between ranks to choose whether or not
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
  CUDACHECK(cudaDeviceGetPCIBusId(info->busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, info->cudaDev));
  for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
    if (info->busId[c] == 0) break;
    info->busId[c] = tolower(info->busId[c]);
  }
  return ncclSuccess;
}

static int getNvlinkCount(const char* busId1, const char* busId2) {
  // Determine if that connection is through NVLink
  int links = 0;
  nvmlDevice_t nvmlDev;
  ncclResult_t res = wrapNvmlDeviceGetHandleByPciBusId(busId1, &nvmlDev);
  if (res != ncclSuccess) return 0;

  for(int l=0; l<NVML_NVLINK_MAX_LINKS; ++l) {
    // nvmlDeviceGetNvLinkState() reports whether a link is enabled or not.
    // Works only on Pascal and later
    nvmlEnableState_t linkState;
    if (wrapNvmlDeviceGetNvLinkState(nvmlDev, l, &linkState) != ncclSuccess) return 0;
    if (linkState == NVML_FEATURE_DISABLED) continue;

    // nvmlDeviceGetNvLinkCapability(NVML_NVLINK_CAP_P2P_SUPPORTED) would seem to
    // report whether the NVLink connects to a peer GPU (versus a POWER CPU?). I
    // don't know whether nvmlDeviceGetNvLinkRemotePciInfo() would succeed in
    // the POWER CPU case, so it seems best to check this as well.
    unsigned canP2P;
    if ((wrapNvmlDeviceGetNvLinkCapability(nvmlDev, l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

    // nvmlDeviceGetNvLinkRemotePciInfo() will return NVML_ERROR_NOT_SUPPORTED
    // if the links don't exist, or are disabled. So checking for that return
    // here would probably make the nvmlDeviceGetNvLinkState check above
    // redundant. Presumably, we still need to check the P2P capability above,
    // since even non-GPUs would posses PCI info.
    nvmlPciInfo_t remoteProc;
    if (wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc) != ncclSuccess) continue;

    if (strncmp(busId2, remoteProc.busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE) == 0) {
      links++;
    }
  }
  return links;
}

/* Determine if we can communicate with the peer */
ncclResult_t p2pCanConnect(int* ret, ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo) {
  struct p2pInfo* myInfo = (struct p2pInfo*)myOpaqueInfo;
  struct p2pInfo* peerInfo = (struct p2pInfo*)peerOpaqueInfo;
  int p2p = 0;
  if (myInfo->hostHash == peerInfo->hostHash) {
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      p2p = 1;
    } else {
      if (cudaDeviceCanAccessPeer(&p2p, myInfo->cudaDev, peerInfo->cudaDev) != cudaSuccess) {
        INFO("peer query failed between dev %d and dev %d",
          myInfo->cudaDev, peerInfo->cudaDev);
        p2p = 0;
      }
      if (p2p == 1) {
        int nlinks = getNvlinkCount(myInfo->busId, peerInfo->busId);
        p2p = nlinks ? 2*nlinks : 1;
      }
    }
  }
  *ret = p2p;
  return ncclSuccess;
}

static int computeRingsRec(int* matrix, int n, int *rings, int currentRing, int nRingsMax, int* inTheRing, int current, int remaining, int connect) {
  int nrings = 0;
  int* line = matrix+current*n;
  inTheRing[current] = 1;
  int currentStep = (currentRing+1)*n-remaining;
  rings[currentStep-1] = current;
  if (remaining == 0) {
    int looprank = rings[currentRing*n];
    if (line[looprank] > 0) {
      if (currentRing+1 == nRingsMax) {
        nrings = 1;
      } else {
	line[looprank]--;
	for (int i=0; i<n; i++) inTheRing[i] = 0;
        if (connect) {
          // First two slots are already set and we need to respect those constraints
          inTheRing[rings[currentStep]] = 1;
	  nrings = 1 + computeRingsRec(matrix, n, rings, currentRing+1, nRingsMax, inTheRing, rings[currentStep+1], n-2, connect);
        } else {
          rings[(currentRing+1)*n] = 0;
	  nrings = 1 + computeRingsRec(matrix, n, rings, currentRing+1, nRingsMax, inTheRing, 0, n-1, connect);
        }
	line[looprank]++;
	for (int i=0; i<n; i++) inTheRing[i] = 1;
      }
    }
  } else {
    int ringsSave[nRingsMax*n];
    int maxStep = 0;
    for (int i=0; i<n; i++) {
      if (inTheRing[i] == 0 && line[i] > 0) {
        line[i]--;
        int nr = computeRingsRec(matrix, n, rings, currentRing, nRingsMax, inTheRing, i, remaining-1, connect);
        if (nr > nrings) {
          nrings = nr;
          maxStep = (nr+currentRing)*n;
          ringsSave[currentStep] = i;
          // Save the rest of the rings
          for (int r=currentStep+1; r<maxStep; r++) {
            ringsSave[r] = rings[r];
          }
          if (nrings + currentRing == nRingsMax) {
            // We found an optimal solution. Let's stop there.
            break;
          }
        }
        line[i]++;
      }
    }
    for (int r=currentStep; r<maxStep; r++) {
      rings[r] = ringsSave[r];
    }
  }
  inTheRing[current] = 0;
  return nrings;
}

int p2pComputeRings(int* matrix, int nranks, int *rings, int nringsMax, int connect) {
  int* inTheRing = (int*)malloc(sizeof(int)*nranks);
  for (int i=0; i<nranks; i++) inTheRing[i] = 0;
  int nrings;
  if (connect) {
    inTheRing[rings[0]] = 1;
    nrings = computeRingsRec(matrix, nranks, rings, 0, nringsMax, inTheRing, rings[1], nranks-2, connect);
  } else {
    rings[0] = 0;
    nrings = computeRingsRec(matrix, nranks, rings, 0, nringsMax, inTheRing, 0, nranks-1, connect);
  }
  free(inTheRing);
  return nrings;
}

static inline int findConnect(int nranks, int* ranks) {
  for (int i = 0; i<nranks; i++) {
    if (ranks[i] != -1) return i;
  }
  return -1;
}

int p2pComputeRingsFromPrevNext(int* values, int nranks, int* rings, int nrings, int* prev, int* next, int oversubscribe, int* nthreads) {
  // Find existing constraints / connections
  int connect = 0;
  for (int r=0; r<nrings; r++) {
    int start = findConnect(nranks, prev+r*nranks);
    int end = findConnect(nranks, next+r*nranks);
    if (start != -1 && end != -1) {
      rings[r*nranks] = end;
      rings[r*nranks+1] = start;
      connect = 1;
    }
  }
  // Compute rings
  int matrix[nranks*nranks];
  for (int i=0; i<nranks; i++) for (int j=0; j<nranks; j++)
    matrix[i*nranks+j] = oversubscribe ? values[i*nranks+j] : values[i*nranks+j]/2;

  int nringsRet = p2pComputeRings(matrix, nranks, rings, nrings, connect);
  if (connect == 0) {
    // Duplicate the rings for NVLink alone
    for (int r=0; r<nringsRet; r++) {
      for (int i=0; i<nranks; i++) rings[(r+nringsRet)*nranks+i] = rings[r*nranks+i];
    }
    nringsRet *= 2;
    *nthreads = 128;
  }
  return nringsRet;
}

ncclResult_t p2pGetRings(int nranks, int* groups, int* subgroups, int* values, int* nringsRet, int* prev, int* next, int minScore, int* nthreads) {
  if (*nringsRet == 0) return ncclSuccess;
  int rings[MAXRINGS*nranks];
  for (int i=0; i<MAXRINGS*nranks; i++) rings[i] = -1;

  // Get the maximum number of rings given the number of nvlinks
  int nrings = MAXRINGS;
  for (int rank=0; rank<nranks; rank++) {
    int nr = 0;
    for (int i=0; i<nranks; i++) {
      nr+= values[rank*nranks+i]/2;
    }
    nrings = min(nrings, nr);
  }
  nrings = min(nrings, *nringsRet);

  if (nrings) {
    int comp_nrings = p2pComputeRingsFromPrevNext(values, nranks, rings, nrings, prev, next, 0, nthreads);
    if (comp_nrings && comp_nrings < nrings*2 && nranks <= 4) {
      // Try to oversubscribe to get a better result
      int rings2[MAXRINGS*nranks];
      for (int i=0; i<MAXRINGS*nranks; i++) rings2[i] = -1;
      int comp2_nrings = p2pComputeRingsFromPrevNext(values, nranks, rings2, nrings*2, prev, next, 1, nthreads);
      if (comp2_nrings > comp_nrings*2) {
        // Oversubscription worked.
        for (int i=0; i<comp2_nrings*nranks; i++) rings[i] = rings2[i];
        comp_nrings = comp2_nrings;
      }
    }
    nrings = comp_nrings;
  }

  int pcie = (nrings == 0) ? 1 : 0;
  if (pcie) {
    // PCIe or QPI
    int connect = 0;
    for (int r=0; r<*nringsRet; r++) {
      int start = findConnect(nranks, prev+r*nranks);
      int end = findConnect(nranks, next+r*nranks);
      if (start != -1 && end != -1) {
        rings[r*nranks] = end;
        rings[r*nranks+1] = start;
        int cur = start;
        int inc = r%2 == 0 ? 1 : -1;
        for (int i=2; i<nranks; i++) {
          cur = (cur+inc+nranks) % nranks;
          while (cur == end || cur == start) cur = (cur+inc+nranks) % nranks;
          rings[r*nranks+i] = cur;
        }
        connect = 1;
      } else {
        if (connect == 1 && r > 0) {
          WARN("Connecting rings but did not find start/end for ring %d. Disabling other rings.", r);
          nrings = *nringsRet = r;
        } else {
          // No connections here. stop.
          break;
        }
      }
    }

    if (connect == 0) {
      nrings = 1;
      for (int r=0; r<nrings; r++) {
        for (int i=0; i<nranks; i++) {
          if (r % 2 == 0)
            rings[r*nranks+i] = i;
          else
            rings[r*nranks+i] = nranks-1-i;
        }
      }
    } else {
      nrings = *nringsRet;
    }
  }

  *nringsRet = nrings;
  for (int ring = 0; ring<nrings; ring++) {
    for (int index=0; index<nranks; index++) {
      int prevIndex = (index - 1 + nranks) % nranks;
      int nextIndex = (index + 1) % nranks;
      int curRank = rings[ring*nranks+index];
      int prevRank = rings[ring*nranks+prevIndex];
      int nextRank = rings[ring*nranks+nextIndex];
      if (prev[ring*nranks+curRank] == -1) prev[ring*nranks+curRank] = prevRank;
      if (next[ring*nranks+curRank] == -1) next[ring*nranks+curRank] = nextRank;
    }
  }
  
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct p2pInfo* myInfo = (struct p2pInfo*)myOpaqueInfo;
  struct p2pInfo* peerInfo = (struct p2pInfo*)peerOpaqueInfo;
  struct p2pConnectInfo info;
  if (myInfo->pid == peerInfo->pid) {
    info.direct = 1;
    info.directPtr = ring->devMem;
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      INFO("%d -> %d via P2P/common device", myInfo->rank, peerInfo->rank);
    } else {
      // Enable P2P access
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d: %s",
            peerInfo->cudaDev, cudaGetErrorString(err));
        return ncclInternalError;
      }
      INFO("%d -> %d via P2P/direct pointer", myInfo->rank, peerInfo->rank);
    }
  } else {
    info.direct = 0;
    // Map IPC and enable P2P access
    if (cudaIpcGetMemHandle(&info.devIpc, (void*)ring->devMem) != cudaSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d", myInfo->rank, peerInfo->cudaDev);
      return ncclInternalError;
    }
    INFO("%d -> %d via P2P/IPC", myInfo->rank, peerInfo->rank);
  }
  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
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
ncclResult_t p2pSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  struct ncclSendRecvMem* remDevMem;
  NCCLCHECK(p2pConnect(connectInfo, send, &remDevMem));
  send->conn.buff = remDevMem->buff;
  send->conn.tail = &remDevMem->tail;
  send->conn.opCount = &remDevMem->opCount;
  // send->conn->head should have been set to devMem already
  return ncclSuccess;
}

ncclResult_t p2pRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  struct ncclSendRecvMem* remDevMem;
  NCCLCHECK(p2pConnect(connectInfo, recv, &remDevMem));
  // recv->conn->buff should have been set to devMem already
  // recv->conn->tail should have been set to devMem already
  // recv->conn->opCount should have been set to devMem already
  recv->conn.head = &remDevMem->head;
  return ncclSuccess;
}

ncclResult_t p2pFree(void* resources) {
  return ncclSuccess;
}

struct ncclTransport p2pTransport = {
  "P2P",
  p2pFillInfo,
  p2pCanConnect,
  p2pGetRings,
  { p2pSetup, p2pSendConnect, p2pFree, NULL },
  { p2pSetup, p2pRecvConnect, p2pFree, NULL }
};


