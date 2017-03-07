/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void print_header() {
  PRINT("# %10s  %12s  %6s  %6s %7s  %5s  %5s  %7s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type", "op",
      "time", "algbw", "busbw", "res", "time", "algbw", "busbw", "res");
}

void print_line_header (int size, int count, const char *typeName, const char *opName, int root) {
  PRINT("%12i  %12i  %6s  %6s", size, count, typeName, opName);
}

void getCollByteCount(size_t *sendbytes, size_t *recvbytes, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t *procSharedBytes, int *sameExpected, size_t nbytes, int nranks) {
    *sendbytes = nbytes*nranks;
    *recvbytes = nbytes;
    *sameExpected = 0;
    *procSharedBytes = nbytes*nranks;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = nbytes;
}

void InitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t recvbytes = args->expectedBytes;
  size_t recvcount = args->expectedBytes / wordSize(type);
  size_t sendbytes = args->sendBytes;
  size_t sendcount = args->sendBytes / wordSize(type);

  while (args->sync[0] != args->thread) pthread_yield();

  for (int i=0; i<args->nGpus; i++) {
    int device;
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];

    if (is_first && i == 0) {
      CUDACHECK(cudaMemcpy(args->procSharedHost, data, sendbytes, cudaMemcpyDeviceToHost));
    } else {
      Accumulate(args->procShared, data, sendcount, type, op);
    }

    CUDACHECK(cudaDeviceSynchronize());
    if (in_place == 0) {
      CUDACHECK(cudaMemset(args->recvbuffs[i], 0, recvbytes));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }

  args->sync[0] = args->thread + 1;

  if (args->thread+1 == args->nThreads) {
#ifdef MPI_SUPPORT
    // Last thread does the MPI reduction
    void* remote, *remoteHost = malloc(sendbytes);
    void* myInitialData = malloc(sendbytes);
    memcpy(myInitialData, args->procSharedHost, sendbytes);
    CUDACHECK(cudaHostRegister(remoteHost, sendbytes, 0));
    CUDACHECK(cudaHostGetDevicePointer(&remote, remoteHost, 0));

    for (int i=0; i<args->nProcs; i++) {
      if (i == args->proc) {
        MPI_Bcast(myInitialData, sendbytes, MPI_BYTE, i, MPI_COMM_WORLD);
        free(myInitialData);
      } else {
        MPI_Bcast(remoteHost, sendbytes, MPI_BYTE, i, MPI_COMM_WORLD);
        Accumulate(args->procShared, remote, sendcount, type, op);
        cudaDeviceSynchronize();
      }
    }
    CUDACHECK(cudaHostUnregister(remoteHost));
    free(remoteHost);
#endif
    args->sync[0] = 0;
  } else {
    while (args->sync[0]) pthread_yield();
  }

  for (int i=0; i<args->nGpus; i++) {
      int offset = ((args->proc*args->nThreads + args->thread)*args->nGpus + i)*recvbytes;
      memcpy(args->expectedHost[i], (void *)((uintptr_t)args->procSharedHost + offset), recvbytes);
  }
}

void GetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = (nranks - 1);
  *busBw = baseBw * factor;
}

void RunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclReduceScatter(sendbuff, recvbuff, count, type, op, comm, stream));
}

void RunTestOp(struct threadArgs_t* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  TimeTest(args, type, typeName, op, opName, 0, 1);
}

void RunTestType(struct threadArgs_t* args, ncclDataType_t type, const char* typeName) {
  RunTestOp(args, type, typeName, ncclSum, "sum");
  RunTestOp(args, type, typeName, ncclProd, "prod");
  RunTestOp(args, type, typeName, ncclMax, "max");
  RunTestOp(args, type, typeName, ncclMin, "min");
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
