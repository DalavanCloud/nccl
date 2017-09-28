/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "env.h"
#include "nvmlwrap.h"
//#include "ibvwrap.h"
#include "rings.h"
#include "bootstrap.h"
#include "transport.h"
#include "common_coll.h"
#include "group.h"
#include "utils.h"
#include "net.h"
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
int ncclChecks;

size_t ncclSingleRingThreshold;

extern "C" __attribute__ ((visibility("default")))
ncclNet_t* ncclNet = NULL;

// We define this as weak to let unit test redefine their own
#pragma weak ncclCudaCompCap
int ncclCudaCompCap() {
  int cudaDev;
  if (cudaGetDevice(&cudaDev) != cudaSuccess) return 0;
  int ccMajor;
  if (cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, cudaDev) != cudaSuccess) return 0;
  return ccMajor;
}

void initNet() {
  if (ncclNet != NULL) {
    INFO("Using external Network %s", ncclNetName());
  } else {
    char* str = getenv("NCCL_IB_DISABLE");
    int ibEnabled = (str && (atoi(str) == 1)) ? 0 : 1;
    ncclNet = ibEnabled && ncclIbSupport() ? &ncclNetIb : &ncclNetSocket;
    INFO("Using internal Network %s", ncclNetName());
  }
}

int ncclLLThreshold;
void initLl() {
  char* str = getenv("NCCL_LL_THRESHOLD");
  ncclLLThreshold = (str && atoi(str) >= 0) ? atoi(str) : NCCL_LL_THRESHOLD;
  INFO("Using NCCL Low-latency algorithm for sizes below %d", ncclLLThreshold);
}

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static ncclResult_t ncclInit() {
  if (initialized) return ncclSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    initEnv();
    initDebug();
    initNet();
    initLl();
    initialized = true;
  }
  pthread_mutex_unlock(&initLock);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  return bootstrapGetUniqueId(out);
}

static ncclResult_t commFree(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  CUDACHECK(cudaFree(comm->devComm));

  for (int ring=0; ring<comm->nRings; ring++) {
    NCCLCHECK(comm->rings[ring].send.transport->send.free(comm->rings[ring].send.transportResources));
    NCCLCHECK(transportDestroyProxy(&comm->rings[ring].send));
    NCCLCHECK(comm->rings[ring].recv.transport->recv.free(comm->rings[ring].recv.transportResources));
    NCCLCHECK(transportDestroyProxy(&comm->rings[ring].recv));
    CUDACHECK(cudaFree(comm->rings[ring].devMem));
    free(comm->rings[ring].userRanks);
    CUDACHECK(cudaFree(comm->rings[ring].devUserRanks));
  }

  if (comm->doneEvent != NULL)
    CUDACHECK(cudaEventDestroy(comm->doneEvent));

  free(comm);
  return ncclSuccess;
}

static ncclResult_t commAlloc(ncclComm_t* comret, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
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
  comm->llThreshold = ncclLLThreshold;

  comm->argsptr = &comm->args;

  *comret = comm;
  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  // Fully duplicate the comm on the device
  CUDACHECK(cudaMalloc(&comm->devComm, sizeof(struct ncclComm)));
  // Copy the comm on the device
  CUDACHECK(cudaMemcpy(comm->devComm, comm, sizeof(struct ncclComm), cudaMemcpyHostToDevice));
  // Copy userRanks
  for (int r=0; r<comm->nRings; r++) {
    CUDACHECK(cudaMemcpy(comm->rings[r].devUserRanks, comm->rings[r].userRanks, comm->nRanks*sizeof(int), cudaMemcpyHostToDevice));
  }
  return ncclSuccess;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#define STR2(v) #v
#define STR(v) STR2(v)
static void showVersion() {
  static int shown = 0;
  if (shown == 0 && ncclDebugLevel >= VERSION) {
    printf("NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX
           "+cuda" STR(CUDA_MAJOR) "." STR(CUDA_MINOR) "\n");
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
  ring->recv.conn.buff = mem->buff;
  ring->recv.conn.llBuff = mem->llBuff;
  ring->recv.conn.tail = &mem->tail;
  ring->recv.conn.opCount = &mem->opCount;
  ring->recv.conn.direct = 0;
  ring->send.conn.head = &mem->head;
  ring->send.conn.llHead = &mem->llHead;
  ring->send.conn.direct = 0;
  ring->send.conn.llStep = 0;
  ring->send.conn.llLastCleaning = 0;
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
  INFO("%s", line);
  for (int i=0; i<nranks; i++) {
    memset(line, ' ', STRLENGTH);
    sprintf(line, "%3d ", i);
    for (int j=0; j<nranks && j<MAXWIDTH; j++) sprintf(4+line+4*j, " %3d", connectMatrix[i*nranks+j]);
    INFO("%s", line);
  }
}

void dumpLine(int* values, int nranks, const char* prefix) {
  int prefixlen = strlen(prefix);
  char line[STRLENGTH+1];
  line[STRLENGTH] = '\0';
  memset(line, ' ', STRLENGTH);
  strncpy(line, prefix, PREFIXLEN);
  for (int i=0; i<nranks && i<MAXWIDTH; i++) sprintf(line+prefixlen+4*i, " %3d", values[i]);
  INFO("%s", line);
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
    if (rank == 0) dumpLine(rings+r*nranks, nranks, prefix);
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

void* waitForNonNullPtr(void* p) {
  volatile void** ptr = (volatile void**) p;
  while (*ptr == NULL) sched_yield();
  return (void*)*ptr;
}

// Allocate/Set Intra Structures and set CG options
ncclResult_t ncclCommSetIntra(struct ncclComm* comm, int rank, int ranks, struct ncclComm* comm0) {
  comm->intraRank = rank;
  comm->intraRanks = ranks;
  comm->intraPhase = 0;

  // Alloc shared structures
  if (comm == comm0) {
    int* bar = (int*)malloc(2*sizeof(int));
    bar[0] = bar[1] = 0;
    comm->intraBarrier = bar;
    comm->intraParams = (struct cudaLaunchParams*)malloc(sizeof(struct cudaLaunchParams)*comm->intraRanks);
    comm->intraCudaDevs = (int*)malloc(sizeof(int)*comm->intraRanks);
    int* CGMode = (int*)malloc(sizeof(int));
    *CGMode = 0x11;
    comm->intraCGMode = CGMode;
  } else {
    comm->intraBarrier = (int*)waitForNonNullPtr(&comm0->intraBarrier);
    comm->intraParams = (struct cudaLaunchParams*)waitForNonNullPtr(&comm0->intraParams);
    comm->intraCudaDevs = (int*)waitForNonNullPtr(&comm0->intraCudaDevs);
    comm->intraCGMode = (int*)waitForNonNullPtr(&comm0->intraCGMode);
  }
  comm->intraCudaDevs[comm->intraRank] = comm->cudaDev;

  // Set CG Mode
  comm->launchMode = ncclComm::GROUP;
  char* str = getenv("NCCL_LAUNCH_MODE");
  if (comm->intraRanks == 1 || (str && strcmp(str, "PARALLEL") == 0)) {
    comm->launchMode = ncclComm::PARALLEL;
  }
  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(cudaStreamCreateWithFlags(&comm->ncclStream, cudaStreamNonBlocking));
  }

  int cgMdLaunch = 0;
#if __CUDACC_VER_MAJOR__ >= 9
  // Check whether the GPU supports Cooperative Group Multi Device Launch
  (void) cudaDeviceGetAttribute(&cgMdLaunch, cudaDevAttrCooperativeMultiDeviceLaunch, comm->cudaDev);
#endif

  // intraCGMode is equal to 1 by default, unless at least one device disables it.
  if (cgMdLaunch == 0) {
    *comm->intraCGMode = 0x10;
  }
  return ncclSuccess;
}

static ncclResult_t initTransportsRank(struct ncclComm* comm, ncclUniqueId* commId) {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  void* commState;
  NCCLCHECK(bootstrapInit(commId, rank, nranks, &commState));
  
  struct ncclInfo* allInfo = (struct ncclInfo*)malloc(sizeof(struct ncclInfo)*nranks);
  NCCLCHECK(fillInfo(allInfo+rank, rank));
  NCCLCHECK(bootstrapAllGather(commState, allInfo, sizeof(struct ncclInfo)));
  int connectTransport[nranks*nranks];
  int connectValue[nranks*nranks];
  NCCLCHECK(fillConnect(allInfo, nranks, rank, connectTransport+nranks*rank, connectValue+nranks*rank));
  NCCLCHECK(bootstrapAllGather(commState, connectTransport, nranks*(sizeof(int))));
  NCCLCHECK(bootstrapAllGather(commState, connectValue, nranks*(sizeof(int))));
  //if (rank == 0) dumpMatrix(connectTransport, nranks);
  //if (rank == 0) dumpMatrix(connectValue, nranks);

  // Get my rings
  int nrings;
  int prev[nranks*MAXRINGS];
  int next[nranks*MAXRINGS];
  comm->nThreads = getDefaultThreads();
  NCCLCHECK(ncclGetRings(&nrings, &comm->nThreads, rank, nranks, connectTransport, connectValue, prev, next));

  // Find max nThreads
  int allData[nranks];
  allData[rank] = comm->nThreads;
  NCCLCHECK(bootstrapAllGather(commState, allData, sizeof(int)));
  for (int i=0; i<nranks; i++)
    comm->nThreads = max(allData[i], comm->nThreads);
  if (rank == 0) INFO("Using %d threads", comm->nThreads);

  // Determine the minimum CUDA Compute capability of all GPUs
  int myCompCap = ncclCudaCompCap();
  int minCompCap = myCompCap;
  allData[rank] = myCompCap;
  NCCLCHECK(bootstrapAllGather(commState, allData, sizeof(int)));
  for (int i=0; i<nranks; i++)
    minCompCap = min(allData[i], minCompCap);
  if (rank == 0) INFO("Min Comp Cap %d", minCompCap);

  // Query the NCCL_SINGLE_RING_THRESHOLD env var
  ncclSingleRingThreshold = getRingThreshold(rank, minCompCap);

  // Find min nrings across ranks
  allData[rank] = nrings;
  NCCLCHECK(bootstrapAllGather(commState, allData, sizeof(int)));
  for (int i=0; i<nranks; i++)
    nrings = min(allData[i], nrings);

  // Exchange data with others to build complete rings
  comm->nRings = nrings;
  for (int r=0; r<nrings; r++) {
    NCCLCHECK(bootstrapAllGather(commState, prev+r*nranks, sizeof(int)));
    NCCLCHECK(bootstrapAllGather(commState, next+r*nranks, sizeof(int)));
  }
  int rings[nranks*MAXRINGS];
  NCCLCHECK(buildRings(nrings, rings, rank, nranks, prev, next));

  // Connect with prev/next for each ring
  for (int r=0; r<nrings; r++) {
    int* ringRanks = rings+r*nranks;
    struct ncclRing *ring = comm->rings+r;
    struct ncclConnect connect[2];
    NCCLCHECK(setupRing(comm, ring, r, rank, nranks, ringRanks, allInfo, connect));
    NCCLCHECK(bootstrapRingExchange(commState, connect, ring->userRanks[nranks-1], ring->userRanks[1], sizeof(struct ncclConnect)));
    NCCLCHECK(ring->send.transport->send.connect(connect+1, &ring->send));
    NCCLCHECK(ring->recv.transport->recv.connect(connect+0, &ring->recv));
  }
  free(allInfo);

  // Intra-process barrier setup
  struct rankInfo {
    uint64_t hostHash;
    int pid;
    struct ncclComm* comm;
  } rankInfos[nranks];
  rankInfos[rank].pid = getpid();
  char hostname[1024];
  getHostName(hostname, 1024);
  rankInfos[rank].hostHash=getHostHash(hostname);
  rankInfos[rank].comm = comm;
  NCCLCHECK(bootstrapAllGather(commState, rankInfos, sizeof(struct rankInfo)));

  // Compute intra ranks
  int intraRank0 = -1, intraRank = -1, intraRanks = 0;
  for (int r=0; r<nranks; r++) {
    if ((rankInfos[r].hostHash == rankInfos[rank].hostHash) && 
        (rankInfos[r].pid == rankInfos[rank].pid)) {
      if (intraRanks == 0) intraRank0 = r;
      if (r == rank) intraRank = intraRanks;
      intraRanks++;
    }
  }
  NCCLCHECK(ncclCommSetIntra(comm, intraRank, intraRanks, rankInfos[intraRank0].comm));

  // Barrier
  bootstrapClose(commState);
  return ncclSuccess;
}

bool SetCpuAffinity(int cudaDev, nvmlDevice_t* nvmlDevice) {
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  if (cudaDeviceGetPCIBusId(busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, cudaDev) != cudaSuccess) return false;
  if (wrapNvmlDeviceGetHandleByPciBusId(busId, nvmlDevice) != ncclSuccess) return false;
  if (wrapNvmlDeviceSetCpuAffinity(*nvmlDevice) != ncclSuccess) return false;
  return true;
}

ncclResult_t ncclCommInitRankSync(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank) {
  NCCLCHECK(wrapNvmlSymbols());
  NCCLCHECK(wrapNvmlInit());

  // Make sure all host memory allocation are close to the GPU
  int cudaDev;
  nvmlDevice_t nvmlDevice;
  CUDACHECK(cudaGetDevice(&cudaDev));
  bool affinity_set = SetCpuAffinity(cudaDev, &nvmlDevice);

  NCCLCHECK(commAlloc(newcomm, ndev, myrank));
  NCCLCHECK(initTransportsRank(*newcomm, &commId));
  NCCLCHECK(devCommSetup(*newcomm));

  if (affinity_set)
    wrapNvmlDeviceClearCpuAffinity(nvmlDevice); // Ignore errors

  NCCLCHECK(wrapNvmlShutdown());
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank) {
  NCCLCHECK(ncclInit());
  if (myrank == 0) showVersion();

  // It seems we need to call this so that NVML doesn't crash later with error
  // 999.
  CUDACHECK(cudaFree(NULL));

  NCCLCHECK(PtrCheck(newcomm, "CommInitRank", "newcomm"));
  if (ndev < 1) {
    WARN("Invalid device count requested : %d", ndev);
    return ncclInvalidArgument;
  }

  if (ncclAsyncMode()) {
    int cudaDev;
    CUDACHECK(cudaGetDevice(&cudaDev));
    return ncclAsyncInit(ncclCommInitRankSync, cudaDev, newcomm, ndev, commId, myrank);
  } else {
    return ncclCommInitRankSync(newcomm, ndev, commId, myrank);
  }
}

static ncclResult_t initTransportsAll(struct ncclComm** comms, const int* devs, int nranks) {
  struct ncclInfo* allInfo = (struct ncclInfo*)malloc(sizeof(struct ncclInfo)*nranks);
  for (int rank=0; rank<nranks; rank++) {
    cudaSetDevice(devs[rank]);
    NCCLCHECK(fillInfo(allInfo+rank, rank));
  }

  int connectTransport[nranks*nranks];
  int connectValue[nranks*nranks];
  for (int rank=0; rank<nranks; rank++)
    NCCLCHECK(fillConnect(allInfo, nranks, rank, connectTransport+nranks*rank, connectValue+nranks*rank));
  
  int prev[nranks*MAXRINGS];
  int prevFinal[nranks*MAXRINGS];
  int next[nranks*MAXRINGS];
  int nextFinal[nranks*MAXRINGS];
  int nrings = MAXRINGS;
  int nthreads=0;
  int myCompCap = ncclCudaCompCap();
  int minCompCap = myCompCap;
  for (int rank=0; rank<nranks; rank++) {
    cudaSetDevice(devs[rank]);
    int nringsRank;
    int nthreadsRank = getDefaultThreads();
    myCompCap = ncclCudaCompCap();
    NCCLCHECK(ncclGetRings(&nringsRank, &nthreadsRank, rank, nranks, connectTransport, connectValue, prev, next));
    nrings = min(nrings, nringsRank);
    nthreads = max(nthreads, nthreadsRank);
    minCompCap = min(minCompCap, myCompCap);
    for (int ring=0; ring<nrings; ring++) {
      int index = ring*nranks+rank;
      prevFinal[index] = prev[index];
      nextFinal[index] = next[index];
    }
  }

  INFO("Using %d threads", nthreads);
  INFO("Min Comp Cap %d", minCompCap);

  // Query the NCCL_SINGLE_RING_THRESHOLD env var
  ncclSingleRingThreshold = getRingThreshold(0, minCompCap);

  int rings[nranks*MAXRINGS];
  NCCLCHECK(buildRings(nrings, rings, 0, nranks, prevFinal, nextFinal));

  for (int rank=0; rank<nranks; rank++) {
    comms[rank]->nRings = nrings;
    comms[rank]->nThreads = nthreads;
  }

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
      NCCLCHECK(ring->send.transport->send.connect(connect+2*rank+1, &ring->send));
      NCCLCHECK(ring->recv.transport->recv.connect(connect+2*rank+0, &ring->recv));
    }
  }
  free(allInfo);
  return ncclSuccess;
}


NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(wrapNvmlSymbols());
  NCCLCHECK(wrapNvmlInit());
  showVersion();

  NCCLCHECK(PtrCheck(comms, "CommInitAll", "comms"));
  int devcount=0;
  cudaGetDeviceCount(&devcount);
  if (ndev < 1 || ndev > devcount) {
    WARN("Invalid device count requested : %d", ndev);
    return ncclInvalidArgument;
  }

  ncclResult_t res;
  int savedDevice;
  int rank, cudaDev;
  ncclComm_t comm = NULL;
  nvmlDevice_t nvmlDevice;
  bool affinity_set = false;
  int ncclDevList[ndev];
  for (int i=0; i<ndev; i++) {
    ncclDevList[i] = devlist ? devlist[i] : i;
  }

  cudaGetDevice(&savedDevice);

  for(rank=0; rank<ndev; ++rank)
    comms[rank] = NULL;

  for (rank=0; rank<ndev; ++rank) {
    cudaDev = ncclDevList[rank];
    if (cudaSetDevice(cudaDev) != cudaSuccess) {
      WARN("rank %d failed to set cuda device %d", rank, cudaDev);
      res = ncclUnhandledCudaError;
      goto cleanup;
    }

    // Set CPU affinity
    affinity_set = SetCpuAffinity(cudaDev, &nvmlDevice);

    res = commAlloc(&comm, ndev, rank);
    if (res != ncclSuccess) {
      WARN("rank %d failed to allocate communicator", rank);
      goto cleanup;
    }
    comms[rank] = comm;

    NCCLCHECK(ncclCommSetIntra(comm, rank, ndev, comms[0]));

    if (affinity_set)
      wrapNvmlDeviceClearCpuAffinity(nvmlDevice); // Ignore errors
  }

  res = initTransportsAll(comms, ncclDevList, ndev);
  if (res != ncclSuccess) {
    WARN("failed to init transports");
    goto cleanup;
  }

  for(rank=0; rank<ndev; ++rank) {
    cudaDev = ncclDevList[rank];
    if (cudaSetDevice(cudaDev) != cudaSuccess) {
      WARN("rank %d failed to set cuda device %d", rank, cudaDev);
      res = ncclUnhandledCudaError;
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

  if (comm->intraRank == 0) {
    free(comm->intraBarrier);
    free(comm->intraParams);
    free(comm->intraCudaDevs);
    free(comm->intraCGMode);
  }

  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(cudaStreamDestroy(comm->ncclStream));
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
  case ncclSystemError            : return "unhandled system error";
  case ncclInternalError          : return "internal error";
  case ncclInvalidArgument        : return "invalid argument";
  case ncclInvalidUsage           : return "invalid usage";
  }
  return "unknown result code";
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NCCLCHECK(PtrCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));
  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NCCLCHECK(PtrCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));
  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NCCLCHECK(PtrCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));
  *rank = comm->rank;
  return ncclSuccess;
}

