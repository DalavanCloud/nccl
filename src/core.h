/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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


#define MAXRINGS 12
#define MAXFLAGS 16
#define MAXRANKS 32
#define DEFAULT_BUFFER_SIZE_BYTES (1UL << 25)
#define NCCL_MEM_PAD_ALIGN 65536


struct ncclMem {
  union { // Pad this block so that devBuff is correctly aligned
    struct {
      int   flags[MAXFLAGS];
      void* recvPtrs[MAXFLAGS];
      int   opCounter; // Used to determine when remote Communicators are ready.
                       // Only used in host memory.
    };
    char pad[NCCL_MEM_PAD_ALIGN];
  };
  // devBuff will be bigger ; we only use its offset/address.
  char buff[1];
};

template <typename T>
struct alignas(long long) DevRing {
  volatile int* __restrict__ prevOpCounter;
  volatile int* __restrict__ nextOpCounter;
  volatile int* __restrict__ sendFlagToNext;
  volatile int* __restrict__ sendFlagToPrev;
  volatile int* __restrict__ recvFlagFromNext;
  volatile int* __restrict__ recvFlagFromPrev;

  T* volatile * __restrict__ recvPtrFromNext;
  T* volatile * __restrict__ sendPtrToPrev;
  T*   __restrict__ recvBuffer;
  T*   __restrict__ sendBuffer;

  int userRank[MAXRANKS];
};

struct NodeRef {
  ncclMem* remote; // TODO: Verify if these
  ncclMem* local;  //       are still needed.
  enum {DEVICE, HOST} type;
  ncclMem* devCleanup;  // Used only when remote comm uses same process & GPU
  ncclMem* hostCleanup; // Used whenever target is in different process
  int* opCounter; // TODO: see if this can be removed too.
};


struct ncclComm {
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // cuda device index
  int nRings;  // number of hamiltonian cycles

  // Device and Host allocated chunks. Stored here to correctly free() memory.
  ncclMem* devMem;
  ncclMem* hostMem;
  int hostMemState;
  int opSched; // Scheduling operation index
  int* opCounter; // Counter of completed operations

  cudaStream_t prevStream; // cache last used stream
  cudaEvent_t doneEvent; // orders operations in different streams

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userFromRing[MAXRINGS];

  // copy of the above stored on each device
  int* devUserFromRing[MAXRINGS];

  // Ring orders
  int* ncclFromRing[MAXRINGS]; // TODO: REMOVE IF NOT NEEDED BEYOND CORE.CU

  // Size of temp buffer in bytes.
  size_t buffSize;
  size_t buffSizePerRing;

  // Whether we have remote access to the recvbuff pointers passed from remote
  // GPUs. In single process mode this can be used as long as QPI links are
  // not present. In multi-process, we never push to a remote recvbuff.
  int globalMemSpace;

  // P2P type : PCIe or NVLink
  enum {PCIE, NVLINK} p2ptype;

  // Device copy of the communicator
  struct ncclComm *devComm;  // TODO: Remove this if not useful

  // Device-side ring views
  DevRing<char>* devRing;

  // Device-to-device communication structures to access remote or local device
  // memory. Actual allocation larger than 1.
  NodeRef ptrs[1];
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

