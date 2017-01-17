/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "nvmlwrap.h"
#include "rings.h"
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
pthread_mutex_t ncclDebugOutputLock;

int ncclPrintCRCs;

#define MAX_ASYNC_THREADS 128
thread_local pthread_t ncclThreads[MAX_ASYNC_THREADS];
thread_local int ncclThreadIndex = 0;
thread_local bool ncclThreadMode = 0;

NCCL_API(ncclResult_t, ncclGroupStart);
ncclResult_t ncclGroupStart() {
  ncclThreadMode = 1;
  return ncclSuccess;
}

struct ncclInitArgs {
  int cudaDev;
  ncclResult_t ret;
  ncclComm_t* newcomm;
  int ndev;
  ncclUniqueId commId;
  int myrank; 
};

NCCL_API(ncclResult_t, ncclGroupEnd);
ncclResult_t ncclGroupEnd() {
  int done = ncclThreadIndex;
  int doneArray[ncclThreadIndex];
  for (int i=0; i<ncclThreadIndex; i++) doneArray[i] = 0;
  while (done) {
    for (int i=0; i<ncclThreadIndex; i++) {
      struct ncclInitArgs* args;
      if (doneArray[i] == 1) continue;
      int err = pthread_tryjoin_np(ncclThreads[i], (void**)&args);
      if (err == EBUSY) continue;
      if (err != 0) return ncclSystemError;
      if (args->ret != ncclSuccess) return args->ret;
      doneArray[i] = 1;
      done--;
      free(args);
    }
  }
  ncclThreadIndex = 0;
  ncclThreadMode = 0;
  return ncclSuccess;
}

ncclResult_t ncclCommInitRankSync(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank);

void* ncclCommInitRankThread(void* args_) {
  struct ncclInitArgs* args = (struct ncclInitArgs*)args_;
  cudaSetDevice(args->cudaDev);
  args->ret = ncclCommInitRankSync(args->newcomm, args->ndev, args->commId, args->myrank);
  return args;
}

static ncclResult_t ncclCommInitRankAsync(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank) {
  struct ncclInitArgs* args = (struct ncclInitArgs*)malloc(sizeof(struct ncclInitArgs));
  
  CUDACHECK(cudaGetDevice(&args->cudaDev));
  args->newcomm = newcomm;
  args->ndev = ndev;
  memcpy(&args->commId, &commId, sizeof(commId));
  args->myrank = myrank;
  
  SYSCHECK(pthread_create(ncclThreads+ncclThreadIndex, NULL, ncclCommInitRankThread, args), "pthread_create");
  ncclThreadIndex++;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  bootstrapGetUniqueId(out);
  if (out == NULL) {
    WARN("Error : no bootstrap available");
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t commFree(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  for (int ring=0; ring<comm->nRings; ring++) {
    free(comm->rings[ring].userRanks);
    CUDACHECK(cudaFree(comm->rings[ring].devUserRanks));
    NCCLCHECK(comm->rings[ring].send.transport->send.free(comm->rings[ring].send.transportResources));
    NCCLCHECK(transportDestroyProxy(&comm->rings[ring].send));
    NCCLCHECK(comm->rings[ring].recv.transport->recv.free(comm->rings[ring].recv.transportResources));
    NCCLCHECK(transportDestroyProxy(&comm->rings[ring].recv));
  }

  if (comm->doneEvent != NULL)
    CUDACHECK(cudaEventDestroy(comm->doneEvent));

  free(comm);
  return ncclSuccess;
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

  // Try to create a CUDA object right away. If there is something wrong with
  // the device we're on (failure cause #1) , better know it early.
  cudaEvent_t doneEvent;
  CUDACHECK(cudaEventCreateWithFlags(&doneEvent, cudaEventDisableTiming));

  struct ncclComm* comm = (struct ncclComm*)malloc(sizeof(struct ncclComm));
  if (comm == NULL) {
    WARN("comm allocation failed");
    return ncclSystemError;
  }
  memset(comm, 0, sizeof(struct ncclComm));

  comm->rank = rank;
  comm->nRanks = ndev;
  cudaGetDevice(&comm->cudaDev);
  comm->doneEvent = doneEvent;

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
  // Copy userRanks
  for (int r=0; r<comm->nRings; r++) {
    CUDACHECK(cudaMemcpy(comm->rings[r].devUserRanks, comm->rings[r].userRanks, comm->nRanks*sizeof(int), cudaMemcpyHostToDevice));
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

static ncclResult_t fillInfo(struct ncclInfo* info, int rank) {
  for (int t=0; t<NTRANSPORTS; t++) {
    NCCLCHECK(ncclTransports[t].fillInfo(info->tinfo+t, rank));
  }
  return ncclSuccess;
}

template <int type>
static ncclResult_t selectTransport(struct ncclInfo* myInfo, struct ncclInfo* peerInfo, struct ncclConnect* connect, struct ncclTransport** transportRet, struct ncclRing* ring) {
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports+t;
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, myInfo->tinfo+t, peerInfo->tinfo+t));
    if (ret > 0) {
      NCCLCHECK(transportComm->setup(myInfo->tinfo+t, peerInfo->tinfo+t, connect, ring));
      *transportRet = transport;
      return ncclSuccess;
    }
  }
  WARN("No transport found !");
  *transportRet = NULL;
  return ncclInternalError;
}

static ncclResult_t setupSendRecv(struct ncclRing* ring) {
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
  ring->buffSize = buffSize;
  const int size = ring->devMemSize = offsetof(struct ncclSendRecvMem, buff)+buffSize;
  struct ncclSendRecvMem* mem;
  CUDACHECK(cudaMalloc(&mem, size));
  CUDACHECK(cudaMemset(mem, 0, size));
  ring->devMem = mem;
  ring->recv.conn.buff = (char*)&mem->buff;
  ring->recv.conn.tail = &mem->tail;
  ring->recv.conn.opCount = &mem->opCount;
  ring->recv.conn.direct = 0;
  ring->send.conn.head = &mem->head;
  ring->send.conn.direct = 0;
  return ncclSuccess;
}

static ncclResult_t fillConnect(struct ncclInfo* allInfo, int nranks, int rank, int* connectTransport, int* connectValue) {
  for (int r=0; r<nranks; r++) {
    connectTransport[r] = -1;
    for (int t=0; t<NTRANSPORTS; t++) {
      NCCLCHECK(ncclTransports[t].canConnect(connectValue+r, allInfo[rank].tinfo+t, allInfo[r].tinfo+t));
      if (connectValue[r] > 0) {
        connectTransport[r] = t;
        break;
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t setupRing(struct ncclComm* comm, struct ncclRing* ring, int ringid, int rank, int nranks, int* ringRanks, struct ncclInfo* allInfo, struct ncclConnect* connect) { 
  ring->id = ringid;
  // Reorganize ranks to start with rank.
  int shift;
  for (shift = 0; shift<nranks; shift++) {
    if (ringRanks[shift] == rank) {
      ring->rank = shift;
      break;
    }
  }
  CUDACHECK(cudaMalloc(&ring->devUserRanks, nranks*sizeof(int)));
  ring->userRanks = (int*)malloc(nranks*sizeof(int));
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+shift)%nranks];
  }
  int prev = ring->userRanks[nranks-1];
  int next = ring->userRanks[1];

  setupSendRecv(ring);
  NCCLCHECK(selectTransport<0>(allInfo+rank, allInfo+prev, connect+0, &ring->recv.transport, ring));
  NCCLCHECK(selectTransport<1>(allInfo+rank, allInfo+next, connect+1, &ring->send.transport, ring));
  NCCLCHECK(transportCreateProxy(0, ring, comm));
  NCCLCHECK(transportCreateProxy(1, ring, comm));
  return ncclSuccess;
}

static void swap(void* mem1, void* mem2, int size) {
  char tmp[size];
  memcpy(tmp, mem1, size); memcpy(mem1, mem2, size); memcpy(mem2, tmp, size);
}

#define MAXWIDTH 20
#define PREFIXLEN 15
#define STRLENGTH (PREFIXLEN+4*MAXWIDTH)
void dumpMatrix(int* connectMatrix, int nranks) {
  char line[STRLENGTH+1];
  line[STRLENGTH] = '\0';
  memset(line, ' ', STRLENGTH);
  for (int j=0; j<nranks && j<MAXWIDTH; j++) sprintf(4+line+4*j, " %3d", j);
  INFO(line);
  for (int i=0; i<nranks; i++) {
    memset(line, ' ', STRLENGTH);
    sprintf(line, "%3d ", i);
    for (int j=0; j<nranks && j<MAXWIDTH; j++) sprintf(4+line+4*j, " %3d", connectMatrix[i*nranks+j]);
    INFO(line);
  }
}

void dumpLine(int* values, int nranks, const char* prefix) {
  int prefixlen = strlen(prefix);
  char line[STRLENGTH+1];
  line[STRLENGTH] = '\0';
  memset(line, ' ', STRLENGTH);
  strncpy(line, prefix, PREFIXLEN);
  for (int i=0; i<nranks && i<MAXWIDTH; i++) sprintf(line+prefixlen+4*i, " %3d", values[i]);
  INFO(line);
}

static ncclResult_t buildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
  for (int r=0; r<nrings; r++) {
    char prefix[30];
    /*sprintf(prefix, "[%d] Ring %d Prev : ", rank, r);
    dumpLine(prev+r*nranks, nranks, prefix);
    sprintf(prefix, "[%d] Ring %d Next : ", rank, r);
    dumpLine(next+r*nranks, nranks, prefix);*/

    int current = rank;
    for (int i=0; i<nranks; i++) {
      rings[r*nranks+i] = current;
      current = next[r*nranks+current];
    }
    sprintf(prefix, "[%d] Ring %d : ", rank, r);
    dumpLine(rings+r*nranks, nranks, prefix);
    if (current != rank) {
      WARN("Error : ring %d does not loop back to start (%d != %d)", r, current, rank);
      return ncclInternalError;
    }
    // Check that all ranks are there
    for (int i=0; i<nranks; i++) {
      int found = 0;
      for (int j=0; j<nranks; j++) {
        if (rings[r*nranks+j] == i) {
          found = 1;
          break;
        }
      }
      if (found == 0) {
        WARN("Error : ring %d does not contain rank %d", r, i);
        return ncclInternalError;
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t initTransportsRank(struct ncclComm* comm, ncclUniqueId* commId) {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  void* commState;
  struct ncclBootstrap* bootstrap;
  NCCLCHECK(bootstrapInit(commId, rank, nranks, &bootstrap, &commState));
  
  struct ncclInfo* allInfo = (struct ncclInfo*)malloc(sizeof(struct ncclInfo)*nranks);
  fillInfo(allInfo+rank, rank);
  NCCLCHECK(bootstrap->allGather(commState, allInfo, sizeof(struct ncclInfo)));
  int connectTransport[nranks*nranks];
  int connectValue[nranks*nranks];
  NCCLCHECK(fillConnect(allInfo, nranks, rank, connectTransport+nranks*rank, connectValue+nranks*rank));
  NCCLCHECK(bootstrap->allGather(commState, connectTransport, nranks*(sizeof(int))));
  NCCLCHECK(bootstrap->allGather(commState, connectValue, nranks*(sizeof(int))));
  //if (rank == 0) dumpMatrix(connectTransport, nranks);
  //if (rank == 0) dumpMatrix(connectValue, nranks);

  // Get my rings
  int nrings;
  int prev[nranks*MAXRINGS];
  int next[nranks*MAXRINGS];
  NCCLCHECK(ncclGetRings(&nrings, rank, nranks, connectTransport, connectValue, prev, next));

  // Find min nrings across ranks
  int allNrings[nranks];
  allNrings[rank] = nrings;
  NCCLCHECK(bootstrap->allGather(commState, allNrings, sizeof(int)));
  for (int i=0; i<nranks; i++)
    nrings = min(allNrings[i], nrings);

  // Exchange data with others to build complete rings
  comm->nRings = nrings;
  for (int r=0; r<nrings; r++) {
    NCCLCHECK(bootstrap->allGather(commState, prev+r*nranks, sizeof(int)));
    NCCLCHECK(bootstrap->allGather(commState, next+r*nranks, sizeof(int)));
  }
  int rings[nranks*MAXRINGS];
  NCCLCHECK(buildRings(nrings, rings, rank, nranks, prev, next));

  // Connect with prev/next for each ring
  for (int r=0; r<nrings; r++) {
    int* ringRanks = rings+r*nranks;
    struct ncclRing *ring = comm->rings+r;
    struct ncclConnect connect[2];
    NCCLCHECK(setupRing(comm, ring, r, rank, nranks, ringRanks, allInfo, connect));
    NCCLCHECK(bootstrap->ringExchange(commState, connect, ring->userRanks[nranks-1], ring->userRanks[1], sizeof(struct ncclConnect)));
    NCCLCHECK(ring->recv.transport->recv.connect(connect+0, &ring->recv));
    NCCLCHECK(ring->send.transport->send.connect(connect+1, &ring->send));
  }
  free(allInfo);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank) {
  return ncclThreadMode ? 
    ncclCommInitRankAsync(newcomm, ndev, commId, myrank) :
    ncclCommInitRankSync(newcomm, ndev, commId, myrank);
}

ncclResult_t ncclCommInitRankSync(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank) {
  if (myrank == 0) showVersion();

  initDebug();
  ncclResult_t res;

  res = wrapNvmlSymbols();
  if (res != ncclSuccess) {
    WARN("NCCL failed to initialize NVML");
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

static ncclResult_t initTransportsAll(struct ncclComm** comms, const int* devs, int nranks) {
  struct ncclInfo* allInfo = (struct ncclInfo*)malloc(sizeof(struct ncclInfo)*nranks);
  for (int rank=0; rank<nranks; rank++) {
    cudaSetDevice(devs[rank]);
    fillInfo(allInfo+rank, rank);
  }

  int connectTransport[nranks*nranks];
  int connectValue[nranks*nranks];
  for (int rank=0; rank<nranks; rank++)
    NCCLCHECK(fillConnect(allInfo, nranks, rank, connectTransport+nranks*rank, connectValue+nranks*rank));
  
  int nrings;
  int nringsFinal = MAXRINGS;
  int prev[nranks*MAXRINGS];
  int prevFinal[nranks*MAXRINGS];
  int next[nranks*MAXRINGS];
  int nextFinal[nranks*MAXRINGS];
  for (int rank=0; rank<nranks; rank++) {
    NCCLCHECK(ncclGetRings(&nrings, rank, nranks, connectTransport, connectValue, prev, next));
    nringsFinal = min(nrings, nringsFinal);
    for (int ring=0; ring<nrings; ring++) {
      int index = ring*nranks+rank;
      prevFinal[index] = prev[index];
      nextFinal[index] = next[index];
    }
  }
  nrings = nringsFinal;
  int rings[nranks*MAXRINGS];
  NCCLCHECK(buildRings(nrings, rings, 0, nranks, prevFinal, nextFinal));

  for (int rank=0; rank<nranks; rank++)
    comms[rank]->nRings = nrings;

  for (int r=0; r<nrings; r++) {
    struct ncclConnect connect[2*nranks];
    int* ringRanks = rings+r*nranks;
    for (int rank=0; rank<nranks; rank++) {
      CUDACHECK(cudaSetDevice(devs[rank]));
      struct ncclRing *ring = comms[rank]->rings+r;
      NCCLCHECK(setupRing(comms[rank], ring, r, rank, nranks, ringRanks, allInfo, connect+2*rank));
    }
    // RingExchange connect information
    for (int rank=0; rank<nranks; rank++) {
      // Swap rank->prev and prevRank->next
      struct ncclRing *ring = comms[rank]->rings+r;
      int prevRank = ring->userRanks[nranks-1];
      struct ncclConnect* prevRankNextConnect = connect+2*prevRank+1;
      struct ncclConnect* rankPrevConnect = connect+2*rank;
      swap(prevRankNextConnect, rankPrevConnect, sizeof(struct ncclConnect));
    }
    for (int rank=0; rank<nranks; rank++) {
      CUDACHECK(cudaSetDevice(devs[rank]));
      struct ncclRing *ring = comms[rank]->rings+r;
      NCCLCHECK(ring->recv.transport->recv.connect(connect+2*rank+0, &ring->recv));
      NCCLCHECK(ring->send.transport->send.connect(connect+2*rank+1, &ring->send));
    }
  }
  free(allInfo);
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
  int ncclDevList[ndev];
  for (int i=0; i<ndev; i++) {
    ncclDevList[i] = devlist ? devlist[i] : i;
  }

  res = wrapNvmlSymbols();
  if (res != ncclSuccess) {
    WARN("NCCL failed to initialize NVML");
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
    cudaDev = ncclDevList[rank];
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

  res = initTransportsAll(comms, ncclDevList, ndev);
  if (res != ncclSuccess) {
    WARN("failed to init transports");
    return res;
  }

  for(rank=0; rank<ndev; ++rank) {
    cudaDev = ncclDevList[rank];
    if (cudaSetDevice(cudaDev) != cudaSuccess) {
      WARN("rank %d failed to set cuda device %d", rank, cudaDev);
      res = ncclInvalidDeviceIndex;
      goto cleanup;
    }
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

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  int savedDevice;
  CUDACHECK(cudaGetDevice(&savedDevice));
  int commDevice = comm->cudaDev;

  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(commDevice));
  }

  commFree(comm);

  if (savedDevice != commDevice)
    CUDACHECK(cudaSetDevice(savedDevice));

  return ncclSuccess;
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

