/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CORE_H_
#define NCCL_CORE_H_

#include "nccl.h"
#include "transport.h"
#include "debug.h"
#include <cstdio>
#include <cuda_runtime.h>

#if __CUDACC_VER_MAJOR__ < 9
struct cudaLaunchParams
{
  void *func;
  dim3 gridDim;
  dim3 blockDim;
  void **args;
  size_t sharedMem;
  cudaStream_t stream;
};
#endif

#define MAXRINGS 12
#define DEFAULT_BUFFER_SIZE_BYTES (1UL << 22) /* 4MiB */

#define DEFAULT_SINGLE_RING_THRESHOLD (1UL << 17) /* 128KiB - but 256KiB for Volta */

extern size_t ncclSingleRingThreshold;
#define LIMIT_NRINGS(SIZE, NRINGS) ((SIZE) <= ncclSingleRingThreshold ? 1 : (NRINGS))

struct ncclConnInfo {
  char *buff;         // Local for recv, remote for send
  int *tail;          // Local for recv, remote for send
  int *head;          // Local for send, remote for recv
  int *opCount;       // Local for recv, remote for send

  int direct;         // Direct communication
  void **ptrExchange; // Pointer exchange for direct communication

  int *fifo;          // Size fifo for proxy
};

struct ncclConnector {
  struct transportProxyInfo* proxyInfo;
  struct ncclTransport* transport;
  void* transportResources; // Host-side resources
  struct ncclConnInfo conn;
};

#define CACHE_LINE_SIZE 128
#define PAGE_SIZE 4096
#define SIZES_FIFO_SIZE 32

struct ncclSendRecvMem {
  union {
    struct {
      int head;
      char pad1[CACHE_LINE_SIZE-sizeof(int)];
      int tail;
      char pad2[CACHE_LINE_SIZE-sizeof(int)];
      void* ptrExchange;
      char pad3[CACHE_LINE_SIZE-sizeof(int)];
      int opCount;
      char pad4[CACHE_LINE_SIZE-sizeof(int)];
      int sizesFifo[SIZES_FIFO_SIZE];
    };
    char pad5[PAGE_SIZE];
  };
  char buff[1]; // Actually larger than that
};

struct ncclRing {
  int id;
  int nthreads;
  // Per ring resources
  struct ncclSendRecvMem* devMem;   // CUDA-size resources
  int buffSize;
  int devMemSize;    // Keep the size for IPCs
  struct ncclConnector send;
  struct ncclConnector recv;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;
  int* devUserRanks;
};

template<typename T>
struct KernelArgs {
  // general parameters
  int root;
  size_t N;

  // local and remote input, output, and buffer
  const T * __restrict__ ThisInput;
  T * __restrict__ ThisOutput;

  struct ncclComm* comm;
  int nRings;
  int opCount;
};

struct ncclProxyParams {
  int substeps;
  int subchunks;
  int nstepsPerRound;
  int nblocksPerRound;
  size_t size;
  int pattern;
};

struct ncclComm {
  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index

  enum { GROUP, PARALLEL } launchMode;
  cudaStream_t userStream; // User provided stream for the current collective
  cudaStream_t ncclStream; // Group Mode : nccl stream
                           // Parallel mode : prev stream
  cudaEvent_t doneEvent;

  // Counter to make sure collectives match (needed for bcast/reduce
  // where syncs are not symmetric).
  int opCount;

  // Rings for collectives 
  int nRings;
  struct ncclRing rings[MAXRINGS];
  int nThreads;
  
  // Device copy of the communicator
  struct ncclComm *devComm;

  // Intra-process sync
  int intraRank;
  int intraRanks;
  int* intraBarrier;
  int intraPhase;

  // Storage for deferred intra-process launch
  struct cudaLaunchParams * intraParams;
  int* intraCudaDevs;
  int* intraCGMode; // Whether we can use CUDA9 CGMD or not
  struct KernelArgs<void> args;
  void* argsptr;
  struct ncclProxyParams proxyParams;
};

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))
#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

// Check CUDA calls
#define CUDACHECK(cmd) do {                                 \
    cudaError_t e = cmd;                                    \
    if( e != cudaSuccess ) {                                \
        WARN("Cuda failure '%s'", cudaGetErrorString(e)); \
        return ncclUnhandledCudaError;                      \
    }                                                       \
} while(false)

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name) do { \
  int ret = -1; \
  while (ret == -1) { \
    SYSCHECKVAL(call, name, ret); \
    if (ret == -1) { \
      INFO("Got %s, retrying", strerror(errno)); \
    }\
  } \
} while (0);

#define SYSCHECKVAL(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) { \
    WARN("Call to " name " failed : %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (0);


// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO("%s:%d -> %d", __FILE__, __LINE__, res); \
    return res; \
  } \
} while (0);

#ifdef PROFAPI
#define NCCL_API(ret, func, args...)        \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((alias(#func)))          \
    ret p##func (args);                     \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((weak))                  \
    ret func(args)
#else
#define NCCL_API(ret, func, args...)        \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    ret func(args)
#endif // end PROFAPI

int ncclCudaCompCap();

#endif // end include guard

