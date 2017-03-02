/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"


void print_header() {
  PRINT("# %10s  %12s  %6s  %7s  %5s  %5s  %7s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type", 
      "time", "algbw", "busbw", "res", "time", "algbw", "busbw", "res");
}

void print_line_header (int size, int count, const char *typeName, const char *opName) {
  PRINT("%12i  %12i  %6s", size, count, typeName);
}

void AllocateBuffs(void **sendbuff, void **recvbuff, void **expected, void **expectedHost, size_t nbytes, int nranks) {
    static int is_first = 1;
    static void *cached_ptr = NULL;
    static void *cached_hostptr = NULL;

    CUDACHECK(cudaMalloc(sendbuff, nbytes));
    CUDACHECK(cudaMalloc(recvbuff, nbytes*nranks));

    if (is_first) { 
        *expectedHost = malloc(nbytes*nranks);
        CUDACHECK(cudaHostRegister(*expectedHost, nbytes*nranks, 0));
        CUDACHECK(cudaHostGetDevicePointer(expected, *expectedHost, 0)); 
        cached_ptr = *expected;
        cached_hostptr = *expectedHost;
        is_first = 0;
    } else {
        *expected = cached_ptr;
        *expectedHost = cached_hostptr; 
    }
}

void InitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t count = args->nbytes / wordSize(type);
  int proc = args->proc;
  int nThreads = args->nThreads;
  int t = args->thread;
  int nGpus = args->nGpus;
  int nBytes = args->nbytes;

  while (args->sync[0] != args->thread) pthread_yield();

  for (int i=0; i<nGpus; i++) {
    int device;
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];


    CUDACHECK(cudaMemcpy((void *)((uintptr_t)args->expected + ((proc*nThreads + t)*nGpus + i)*count*wordSize(type)), 
                data, 
                count*wordSize(type), cudaMemcpyDeviceToHost));

    if (in_place == 0) {
      CUDACHECK(cudaMemset(args->recvbuffs[i], 0, nBytes));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }

  args->sync[0] = t + 1;

  if (t+1 == nThreads) {
#ifdef MPI
    // Last thread does the MPI allgather
    MPI_Allgather(MPI_IN_PALCE, nBytes*nThreads*nGpus, args->expected, nBytes*nThreads*nGpus, MPI_BYTE, MPI_COMM_WORLD);
#endif
    args->sync[0] = 0;
  } else {
    while (args->sync[0]) pthread_yield();
  }
}

void GetBw(double baseBw, double* algBw, double* busBw, int nranks) {
  *algBw = baseBw;
  double factor = 2 * nranks - 2;
  factor /= nranks;
  *busBw = baseBw * factor;
}

void RunColl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclAllGather(sendbuff, recvbuff, count, type, comm, stream));
}

void RunTestType(struct threadArgs_t* args, ncclDataType_t type, const char* typeName) {
  TimeTest(args, type, typeName, (ncclRedOp_t)0, 0, NULL);
}

void RunTests(struct threadArgs_t* args) {
  RunTestType(args, ncclInt8, "int8");
  RunTestType(args, ncclUint8, "uint8");
  RunTestType(args, ncclInt32, "int32");
  RunTestType(args, ncclUint32, "uint32");
  RunTestType(args, ncclInt64, "int64");
  RunTestType(args, ncclUint64, "uint64");
  RunTestType(args, ncclHalf, "half");
  RunTestType(args, ncclFloat, "float");
  RunTestType(args, ncclDouble, "double");
}
