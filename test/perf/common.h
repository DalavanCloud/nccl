/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include <stdio.h>
#include <algorithm>
#include <curand.h>
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Cuda failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("NCCL failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

struct threadArgs_t {
  size_t nbytes;
  size_t minbytes;
  size_t maxbytes;
  size_t stepbytes;
  size_t stepfactor;

  int nProcs;
  int proc;
  int nThreads;
  int thread;
  int nGpus;
  void** sendbuffs;
  size_t sendBytes;
  size_t sendInplaceOffset;
  void** recvbuffs;
  size_t recvInplaceOffset;
  ncclComm_t* comms;
  cudaStream_t* streams;

  void** expectedHost;
  void** expected;
  size_t expectedBytes;
  void* procSharedHost;
  void* procShared;
  volatile int* sync;
  int syncRank;
  int syncNranks;
  double* deltaHost;
  double* delta;
  int* errors;
  double* bw;
  int* bw_count;
};

#include <chrono>

// Provided by common.cu
extern void TimeTest(struct threadArgs_t* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op,  const char* opName, int root, int inPlace);
extern void Randomize(void* ptr, int count, ncclDataType_t type, int seed);
extern void Accumulate(void* out, void* in, int n, ncclDataType_t type, ncclRedOp_t op);
extern void CheckDelta(void* expected, void* results, int count, ncclDataType_t type, double* devmax);
extern double DeltaMaxValue(ncclDataType_t type);

// Provided by each coll
void RunTest(struct threadArgs_t* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName);
extern void GetBw(size_t count, int typeSize, double sec, double* algBw, double* busBw, int nranks);
extern void RunColl(void* sendbuf, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op,  int root, ncclComm_t comm, cudaStream_t stream);
extern void InitData(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op,  int in_place, int is_first);
extern double CheckData(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op);
extern void AllocateBuffs(void **sendbuff, void **recvbuff, void **expected, void **expectedHost, size_t nbytes, int nranks);
extern void InitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op,  int root, int in_place, int is_first);
extern void getCollByteCount(size_t *sendbytes, size_t *recvbytes, size_t *sendInlineOffset, size_t *recvInlineOffset, size_t *procSharedBytes, int *sameexpected, size_t nbytes, int nranks);
extern void print_line_header (int size, int count, const char *typeName, const char *opName, int root);
extern void print_header();

#include <unistd.h>

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

#include <stdint.h>

static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static size_t wordSize(ncclDataType_t type) {
  switch(type) {
//  case ncclChar:
    case ncclInt8:
    case ncclUint8: return 1;
//  case ncclHalf:
    case ncclFloat16: return 2;
//  case ncclInt:
    case ncclInt32:
    case ncclUint32:
//  case ncclFloat:
    case ncclFloat32: return 4;
    case ncclInt64:
    case ncclUint64:
//  case ncclDouble:
    case ncclFloat64: return 8;
    default: return 0;
  }
}

extern ncclDataType_t test_types[ncclNumTypes];
extern const char *test_typenames[ncclNumTypes];
extern ncclRedOp_t test_ops[ncclNumOps];
extern const char *test_opnames[ncclNumOps];

extern thread_local int is_main_thread;
#define PRINT if (is_main_thread) printf


