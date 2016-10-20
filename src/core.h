/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef CORE_H_
#define CORE_H_

#include "nccl.h"
#include <cstdio>
#include <cuda_runtime.h>

// DIE on error
#define CUDACHECK(cmd) do {                              \
    cudaError_t e = cmd;                                 \
    if( e != cudaSuccess ) {                             \
        printf("Cuda failure %s:%d '%s'\n",              \
               __FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                              \
    }                                                    \
} while(false)

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t res = call; \
  if (call != ncclSuccess) { \
    return res; \
  } \
} while (0);

#define MAXRINGS 12
#define MAXFLAGS (2*MAXRINGS)
#define MAXRANKS 32
#define DEFAULT_BUFFER_SIZE_BYTES (1UL << 21)

struct ncclConnInfo {
  char *buff;         // Local for recv, remote for send
  int *tail;          // Local for recv, remote for send
  int *head;          // Local for send, remote for recv

  int direct;         // Direct communication
  void **ptrExchange; // Pointer exchange for direct communication
};

struct ncclConnector {
  struct ncclTransport* transport;
  void* transportResources; // Host-side resources
  struct ncclConnInfo conn;
};

#define CACHE_LINE_SIZE 128
#define PAGE_SIZE 4096

struct ncclSendRecvMem {
  union {
    struct {
      int head;
      char pad1[CACHE_LINE_SIZE-sizeof(int)];
      int tail;
      char pad2[CACHE_LINE_SIZE-sizeof(int)];
      void* ptrExchange;
      char pad3[CACHE_LINE_SIZE-sizeof(int)];
    };
    char pad4[PAGE_SIZE];
  };
  char buff[1]; // Actually larger than that
};

struct ncclSendRecv {
  struct ncclSendRecvMem* devMem;   // CUDA-size resources
  int devMemSize;    // Keep the size for IPCs
  struct ncclConnector send;
  struct ncclConnector recv;
};

struct ncclRing {
  int rank;
  int id;
  // Per ring resources
  int buffSize;
  struct ncclSendRecv sendrecv;
  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;
};

struct ncclComm {
  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index

  enum { PCIE, NVLINK } p2ptype;

  cudaStream_t prevStream; // cache last used stream
  cudaEvent_t doneEvent; // orders operations in different streams

  // Rings for collectives 
  int nRings;
  struct ncclRing rings[MAXRINGS];
  
  // Device copy of the communicator
  struct ncclComm *devComm;
};


extern int ncclPrintCRCs;

typedef enum {NONE=0, VERSION=1, WARN=2, INFO=3, ABORT=4} DebugLevel;
extern DebugLevel ncclDebugLevel;

#define WARN(...) do {                                           \
  if (ncclDebugLevel >= WARN) {                                  \
    printf("WARN %s:%d ", __FILE__, __LINE__);                   \
    printf(__VA_ARGS__);                                         \
    printf("\n");                                                \
    fflush(stdout);                                              \
    if (ncclDebugLevel >= ABORT) abort();                        \
  }                                                              \
} while(0)

#define INFO(...) do {                                           \
  if (ncclDebugLevel >= INFO) {                                  \
    printf("INFO "); printf(__VA_ARGS__); printf("\n");          \
    fflush(stdout);                                              \
  }                                                              \
} while(0)

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


#endif // end include guard

