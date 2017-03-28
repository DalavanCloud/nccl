/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef COMMON_COLL_H_
#define COMMON_COLL_H_

#include "core.h"

static ncclResult_t PointerCheck(const void* pointer, struct ncclComm* comm, const char* ptrname, const char* opname) {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, pointer);
  if (err != cudaSuccess || attr.devicePointer == NULL) {
    WARN("%s : %s is not a valid pointer", opname, ptrname);
    return ncclInvalidArgument;
  }
  if (attr.memoryType == cudaMemoryTypeDevice && attr.device != comm->cudaDev) {
    WARN("%s : %s allocated on device %d mismatchs with NCCL device %d", opname, ptrname, attr.device, comm->cudaDev);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

static ncclResult_t PtrCheck(void* ptr, const char* opname, const char* ptrname) {
  if (ptr == NULL) {
    WARN("%s : %s argument is NULL", opname, ptrname);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

static ncclResult_t ArgsCheck(const void* sendbuff, const void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, struct ncclComm* comm, const char* opname) {
  NCCLCHECK(PtrCheck(comm, opname, "comm"));
  // First, the easy ones
  if (root < 0 || root >= comm->nRanks) {
    WARN("%s : invalid root %d (root should be in the 0..%d range)", opname, root, comm->nRanks);
    return ncclInvalidArgument;
  }
  if (type < 0 || type >= ncclNumTypes) {
    WARN("%s : invalid type %d", opname, type);
    return ncclInvalidArgument;
  }
  if (op < 0 || op >= ncclNumOps) {
    WARN("%s : invalid reduction operation %d", opname, op);
    return ncclInvalidArgument;
  }

  // Check pointers
  NCCLCHECK(PointerCheck(sendbuff, comm, "sendbuff", opname))
  if (strcmp(opname, "Reduce") == 0 && comm->rank != root) {
    // No need to check recvbuff pointer for non-root reduce
    return ncclSuccess;
  }
  NCCLCHECK(PointerCheck(recvbuff, comm, "recvbuff", opname))
  return ncclSuccess;
}

// Kernel launch
template<typename T>
struct KernelArgs {
  // general parameters
  int root;
  int N;

  // local and remote input, output, and buffer
  const T * __restrict__ ThisInput;
  T * __restrict__ ThisOutput;

  struct ncclComm* comm;
  int nRings;
  int opCount;
};

template<typename T>
void ArgsSetup(KernelArgs<T> *args, const void* sendbuff, void* recvbuff,
		const int root, const size_t count, ncclComm *comm) {
  args->root = root;
  args->N = count;
  args->ThisInput = (const T*)sendbuff;
  args->ThisOutput = (T*)recvbuff;
  args->comm = comm->devComm;
  args->nRings = comm->nRings;
  args->opCount = comm->opCount;
  comm->opCount++;
}

#define LAUNCH_KERNEL(K, threads, UNROLL, FUNC, T, \
		args, stream) do { \
  dim3 grid(args.nRings, 1, 1); \
  dim3 block(threads+1, 1, 1); \
  void* argptrs[] = {&args}; \
  /* Generate code for the 3 possible sizes */ \
  if (threads == 128) { \
    CUDACHECK(cudaLaunchKernel( \
          (void*)K<128, UNROLL, FUNC, T>, \
          grid, block, argptrs, 0, stream)); \
  } else if (threads == 256) { \
    CUDACHECK(cudaLaunchKernel( \
          (void*)K<256, UNROLL, FUNC, T>, \
          grid, block, argptrs, 0, stream)); \
  } else if (threads == 512) { \
    CUDACHECK(cudaLaunchKernel( \
          (void*)K<512, UNROLL, FUNC, T>, \
          grid, block, argptrs, 0, stream)); \
  } else { \
    WARN("Error : forbidden number of threads %d", threads); \
    return ncclInternalError; \
  } \
} while (0)

#endif
