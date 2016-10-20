/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "libwrap.h"
#include "topo.h"
#include "bootstrap.h"
#include "transport.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <string.h>
#include <errno.h>

DebugLevel ncclDebugLevel;
int ncclPrintCRCs;

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  bootstrapGetUniqueId(out);
  if(out == NULL) {
    WARN("Error : no bootstrap available");
    return ncclInternalError;
  }
  return ncclSuccess;
}

static void initDebug() {
  const char* nccl_debug = getenv("NCCL_DEBUG");
  if (nccl_debug == NULL) {
    ncclDebugLevel = NONE;
  } else if (strcmp(nccl_debug, "VERSION") == 0) {
    ncclDebugLevel = VERSION;
  } else if (strcmp(nccl_debug, "WARN") == 0) {
    ncclDebugLevel = WARN;
  } else if (strcmp(nccl_debug, "INFO") == 0) {
    ncclDebugLevel = INFO;
    INFO("NCCL debug level set to INFO");
  } else if (strcmp(nccl_debug, "ABORT") == 0) {
    ncclDebugLevel = ABORT;
    INFO("NCCL debug level set to ABORT");
  }

  const char* nccl_crc = getenv("NCCL_CRC");
  if (nccl_crc != NULL && strcmp(nccl_crc, "PRINT")==0 ) {
    ncclPrintCRCs = 1;
  } else {
    ncclPrintCRCs = 0;
  }
}

static void commFree(ncclComm_t comm) {
  if (comm == NULL)
    return;

  if (comm->doneEvent != NULL)
    if (cudaEventDestroy(comm->doneEvent) != cudaSuccess)
      INFO("ncclComm failed to destroy doneEvent");

  free(comm);
}

static ncclResult_t commAlloc(ncclComm_t* comret, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclUnsupportedDeviceCount;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidRank;
  }

  struct ncclComm* comm = (struct ncclComm*)malloc(sizeof(struct ncclComm));
  if (comm == NULL) {
    WARN("comm allocation failed");
    return ncclSystemError;
  }
  memset(comm, 0, sizeof(struct ncclComm));

  comm->rank = rank;
  comm->nRanks = ndev;
  cudaGetDevice(&comm->cudaDev);

  if (cudaEventCreateWithFlags(&comm->doneEvent, cudaEventDisableTiming) != cudaSuccess) {
    WARN("ncclComm on rank %d failed to create doneEvent", rank);
    commFree(comm);
    return ncclUnhandledCudaError;
  }

  *comret = comm;
  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  // Fully duplicate the comm on the device
  if (cudaMalloc(&comm->devComm, sizeof(struct ncclComm)) != cudaSuccess) {
    WARN("failed to allocated device comm");
    return ncclCudaMallocFailed;
  }
  // Copy the comm on the device
  if (cudaMemcpy(comm->devComm, comm, sizeof(struct ncclComm), cudaMemcpyHostToDevice) != cudaSuccess) {
    WARN("failed to copy device comm");
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

static void showVersion() {
  static int shown = 0;
  if (shown == 0 && ncclDebugLevel >= VERSION) {
    printf("NCCL version %d.%d.%d compiled with CUDA %d.%d\n", NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, CUDA_MAJOR, CUDA_MINOR);
    fflush(stdout);
    shown = 1;
  }
}

static ncclResult_t fillInfo(struct ncclInfo* info) {
  for (int t=0; t<NTRANSPORTS; t++) {
    ncclTransports[t].fillInfo(info->tinfo+t);
  }
  return ncclSuccess;
}

template <int type>
static int selectTransport(struct ncclInfo* myInfo, struct ncclInfo* peerInfo, struct ncclConnect* connect, struct ncclTransport** transport, struct ncclRing* ring) {
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransportComm* transportComm = type == 1 ? &ncclTransports[t].send : &ncclTransports[t].recv;
    if (transportComm->setup(myInfo->tinfo, peerInfo->tinfo, connect, ring) == 0) {
      *transport = ncclTransports+t;
      return 0;
    }
  }
  WARN("No transport found !");
  return 1;
}

static ncclResult_t setupSendRecv(struct ncclSendRecv* sendrecv) {
  const char* str = getenv("NCCL_BUFFSIZE");
  int buffSize;
  if (str != NULL) {
    errno = 0;
    buffSize = strtol(str, NULL, 10);
    if (errno == ERANGE || buffSize == 0) {
      INFO("invalid NCCL_BUFFSIZE: %s, using default %lu",
          str, DEFAULT_BUFFER_SIZE_BYTES);
      buffSize = DEFAULT_BUFFER_SIZE_BYTES;
    }
  } else {
    buffSize = DEFAULT_BUFFER_SIZE_BYTES;
  }
  const int size = sendrecv->devMemSize = sizeof(ncclSendRecvMem)-1+buffSize;
  struct ncclSendRecvMem* mem;
  cudaMalloc(&mem, size);
  sendrecv->devMem = mem;
  sendrecv->recv.conn.buff = (char*)&mem->buff;
  sendrecv->recv.conn.tail = &mem->tail;
  sendrecv->recv.conn.direct = 0;
  sendrecv->send.conn.head = &mem->head;
  sendrecv->send.conn.direct = 0;
  return ncclSuccess;
}

static int getRings(int** rings, int nranks) {
  // TODO : something better !
  *rings = (int*)malloc(sizeof(int)*nranks);
  for (int i=0; i<nranks; i++)
    rings[0][i] = i;
  return 1;
}

static ncclResult_t initTransportsAll(struct ncclComm** comms, const int* devs, int nranks) {
  struct ncclInfo* allInfo = (struct ncclInfo*)malloc(sizeof(struct ncclInfo)*nranks);
  for (int rank=0; rank<nranks; rank++) {
    cudaSetDevice(devs[rank]);
    fillInfo(allInfo+rank);
  }
  
  int *rings;
  int nrings = getRings(&rings, nranks);

  for (int r=0; r<nrings; r++) {
    struct ncclConnect connect[2*nranks];
    for (int rank=0; rank<nranks; rank++) {
      struct ncclRing *ring = comms[rank]->rings+r;
      int *ringRanks = rings+nranks*r;
      int prev = ringRanks[nranks-1], next = ringRanks[1];
      for (int i=0; i<nranks-1; i++) {
        if (ringRanks[i+1] == rank) break;   
        prev = ringRanks[i]; 
        next = ringRanks[i+2];
      }
      setupSendRecv(&ring->sendrecv);
      selectTransport<0>(allInfo+rank, allInfo+prev, connect+2*prev+1, &ring->sendrecv.recv.transport, ring);
      selectTransport<1>(allInfo+rank, allInfo+next, connect+2*next+0, &ring->sendrecv.send.transport, ring);
    }
    for (int rank=0; rank<nranks; rank++) {
      struct ncclRing *ring = comms[rank]->rings+r;
      ring->sendrecv.recv.transport->recv.connect(connect+2*rank+0, &ring->sendrecv.recv);
      ring->sendrecv.send.transport->send.connect(connect+2*rank+1, &ring->sendrecv.send);
    }
  }
  free(rings);
  return ncclSuccess;
}

static ncclResult_t initTransportsRank(struct ncclComm* comm, ncclUniqueId* commId) {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  void* commState;
  struct ncclBootstrap* bootstrap;
  NCCLCHECK(bootstrapInit(commId, rank, nranks, &bootstrap, &commState));
  
  struct ncclInfo* allInfo = (struct ncclInfo*)malloc(sizeof(struct ncclInfo)*nranks);
  fillInfo(allInfo+rank);
  NCCLCHECK(bootstrap->allGather(commState, allInfo, sizeof(struct ncclInfo)));

  int *rings;
  int nrings = getRings(&rings, nranks);

  for (int r=0; r<nrings; r++) {
    struct ncclConnect connect[2];
    struct ncclRing* ring = comm->rings+r;
    int *ringRanks = rings+nranks*r;
    int prev = ringRanks[nranks-1], next = ringRanks[1];
    for (int i=0; i<nranks-1; i++) {
      if (ringRanks[i+1] == rank) break;   
      prev = ringRanks[i]; 
      next = ringRanks[i+2];
    }
    setupSendRecv(&ring->sendrecv);
    selectTransport<0>(allInfo+rank, allInfo+prev, connect+0, &ring->sendrecv.recv.transport, ring);
    selectTransport<1>(allInfo+rank, allInfo+next, connect+1, &ring->sendrecv.send.transport, ring);
    NCCLCHECK(bootstrap->ringExchange(commState, connect, sizeof(struct ncclConnect)));
    ring->sendrecv.recv.transport->recv.connect(connect+0, &ring->sendrecv.recv);
    ring->sendrecv.send.transport->send.connect(connect+1, &ring->sendrecv.send);
  }
  free(rings);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank) {
  if (myrank == 0) showVersion();

  if (strlen(commId.internal) < 1 ||
      strlen(commId.internal) >= NCCL_UNIQUE_ID_BYTES) {
    WARN("rank %d invalid commId", myrank);
    return ncclInvalidArgument;
  }

  initDebug();
  ncclResult_t res;

  res = wrapSymbols();
  if (res != ncclSuccess) {
    WARN("NCCL failed to initialize client libs");
    return res;
  }

  res = wrapNvmlInit();
  if (res != ncclSuccess) {
    WARN("rank %d failed to initialize nvml", myrank);
    return res;
  }

  res = commAlloc(newcomm, ndev, myrank);
  if (res != ncclSuccess) {
    WARN("rank %d failed to allocate communicator", myrank);
    return res;
  }

  res = initTransportsRank(*newcomm, &commId);
  if (res != ncclSuccess) {
    WARN("rank %d failed to init transports", myrank);
    return res;
  }

  res = devCommSetup(*newcomm);
  if (res != ncclSuccess) {
    WARN("rank %d failed to copy dcomm", myrank);
    return res;
  }

  if (wrapNvmlShutdown() != ncclSuccess)
    INFO("rank %d did not shutdown nvml properly", myrank);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  initDebug();

  showVersion();

  ncclResult_t res;
  int savedDevice;
  int rank, cudaDev;
  ncclComm_t comm = NULL;
  char busId[13];
  nvmlDevice_t nvmlHandle;
  int affinity_set = 0;

  res = wrapSymbols();
  if (res != ncclSuccess) {
    WARN("NCCL failed to initialize client libs");
    return res;
  }

  cudaGetDevice(&savedDevice);

  res = wrapNvmlInit();
  if (res != ncclSuccess) {
    WARN("nccl failed to initialize nvml");
    return res;
  }

  for(rank=0; rank<ndev; ++rank)
    comms[rank] = NULL;

  for (rank=0; rank<ndev; ++rank) {
    cudaDev = (devlist == NULL) ? rank : devlist[rank];
    if (cudaSetDevice(cudaDev) != cudaSuccess) {
      WARN("rank %d failed to set cuda device %d", rank, cudaDev);
      res = ncclInvalidDeviceIndex;
      goto cleanup;
    }

    // Set CPU affinity
    affinity_set = 0;
    if (cudaDeviceGetPCIBusId(busId, 13, cudaDev) != cudaSuccess) {
      INFO("rank %d failed to get PCI Bus Id for device %d", rank, cudaDev);
      goto skipaffinity;
    }
    if (wrapNvmlDeviceGetHandleByPciBusId(busId, &nvmlHandle) != ncclSuccess) {
      INFO("rank %d failed to get nvml handle for device %s", rank, busId);
      goto skipaffinity;
    }
    if (wrapNvmlDeviceSetCpuAffinity(nvmlHandle) != ncclSuccess) {
      INFO("rank %d failed to set affinity", rank);
      goto skipaffinity;
    }
    affinity_set = 1;
    skipaffinity:

    res = commAlloc(&comm, ndev, rank);
    if (res != ncclSuccess) {
      WARN("rank %d failed to allocate communicator", rank);
      goto cleanup;
    }
    comms[rank] = comm;

    if (affinity_set && wrapNvmlDeviceClearCpuAffinity(nvmlHandle) != ncclSuccess) {
      INFO("rank %d set but failed to clear cpu affinity", rank);
    }
  }

  res = initTransportsAll(comms, devlist, ndev);
  if (res != ncclSuccess) {
    WARN("failed to init transports");
    return res;
  }

  for(rank=0; rank<ndev; ++rank) {
    res = devCommSetup(comms[rank]);
    if (res != ncclSuccess) {
      WARN("rank %d failed to copy dcomm", rank);
    }
  }

  res = ncclSuccess;
  goto final;

  cleanup:
  for(rank=0; rank<ndev; ++rank) {
    if(comms[rank] != NULL) {
      commFree(comms[rank]);
    }
  }

  final:
  if(wrapNvmlShutdown() != ncclSuccess)
    INFO("NCCL did not shutdown nvml properly");
  cudaSetDevice(savedDevice);
  return res;
}

NCCL_API(void, ncclCommDestroy, ncclComm_t comm);
void ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL)
    return;

  int savedDevice;
  cudaGetDevice(&savedDevice);
  int commDevice = comm->cudaDev;

  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(commDevice));
  }

  commFree(comm);

  if (savedDevice != commDevice)
    cudaSetDevice(savedDevice);
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
  case ncclSuccess                : return "no error";
  case ncclUnhandledCudaError     : return "unhandled cuda error";
  case ncclSystemError            : return "system error";
  case ncclInternalError          : return "internal error";
  case ncclInvalidDevicePointer   : return "invalid device pointer";
  case ncclInvalidRank            : return "invalid rank";
  case ncclUnsupportedDeviceCount : return "unsupported device count";
  case ncclDeviceNotFound         : return "device not found";
  case ncclInvalidDeviceIndex     : return "invalid device index";
  case ncclLibWrapperNotSet       : return "lib wrapper not initialized";
  case ncclCudaMallocFailed       : return "cuda malloc failed";
  case ncclRankMismatch           : return "parameter mismatch between ranks";
  case ncclInvalidArgument        : return "invalid argument";
  case ncclInvalidType            : return "invalid data type";
  case ncclInvalidOperation       : return "invalid reduction operations";
  }
  return "unknown result code";
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  *rank = comm->rank;
  return ncclSuccess;
}

