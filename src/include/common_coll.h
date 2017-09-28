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

template<typename T>
void ArgsSetup(const T* sendbuff, T* recvbuff,
		const int root, const size_t count, ncclComm *comm) {
  struct KernelArgs<void>* args = &comm->args;
  args->root = root;
  args->N = count;
  args->ThisInput = sendbuff;
  args->ThisOutput = recvbuff;
  args->comm = comm->devComm;
  args->opCount = comm->opCount;
}

#define SAVE_KERNEL(K, comm, UNROLL, FUNC, T, stream) do { \
  int nRings = comm->args.nRings = LIMIT_NRINGS(count*sizeof(T), comm->nRings); \
  dim3 grid(nRings, 1, 1); \
  dim3 block(comm->nThreads+1, 1, 1); \
  void* f; \
  /* Generate code for the 3 possible sizes */ \
  if (comm->nThreads == 128) { \
    f=(void*)K<128, UNROLL, FUNC, T>; \
  } else if (comm->nThreads == 256) { \
    f=(void*)K<256, UNROLL, FUNC, T>; \
  } else if (comm->nThreads == 512) { \
    f=(void*)K<512, UNROLL, FUNC, T>; \
  } else { \
    WARN("Error : forbidden number of threads %d", comm->nThreads); \
    return ncclInternalError; \
  } \
  comm->userStream = stream; \
  struct cudaLaunchParams params = { f, grid, block, &comm->argsptr, 0, comm->ncclStream }; \
  memcpy(comm->intraParams+comm->intraRank, &params, sizeof(params)); \
} while (0)

#define SAVE_KERNEL_SMALL(K, comm, FUNC, T, stream) do { \
  dim3 grid(1, 1, 1); \
  dim3 block(LL_NTHREADS, 1, 1); \
  static_assert(LL_NTHREADS*sizeof(union ncclLLFifoLine)*NUM_LL_CHUNKS <= LL_BUFF_SIZE, "LL_BUFF_SIZE is too low."); \
  comm->userStream = stream; \
  void* f = (void*)K<LL_NTHREADS, FUNC, T>; \
  struct cudaLaunchParams params = { f, grid, block, &comm->argsptr, 0, comm->ncclStream }; \
  memcpy(comm->intraParams+comm->intraRank, &params, sizeof(params)); \
} while (0)

#endif
