/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void print_header() {
  PRINT("# %10s  %12s  %6s  %6s        out-of-place                    in-place\n", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %6s  %7s  %5s  %5s  %7s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type", "op",
      "time", "algbw", "busbw", "res", "time", "algbw", "busbw", "res");
}

void print_line_header (int size, int count, const char *typeName, const char *opName, int root) {
  PRINT("%12i  %12i  %6s  %6s", size, count, typeName, opName);
}

void getCollByteCount(size_t *sendbytes, size_t *recvbytes, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t *procSharedBytes, int *sameExpected, size_t nbytes, int nranks) {
    *sendbytes = nbytes;
    *recvbytes = nbytes;
    *sameExpected = 1;
    *procSharedBytes = 0;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
 }

void InitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t count = args->nbytes / wordSize(type);

  while (args->sync[0] != args->thread) pthread_yield();

  for (int i=0; i<args->nGpus; i++) {
    int device;
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];

    if (is_first && i == 0) {
      CUDACHECK(cudaMemcpy(args->expected[0], data, count*wordSize(type), cudaMemcpyDeviceToHost));
    } else {
      Accumulate(args->expected[0], data, count, type, op);
    }

    if (in_place == 0) {
      CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->nbytes));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }

  args->sync[0] = args->thread + 1;

  if (args->thread+1 == args->nThreads) {
#ifdef MPI_SUPPORT
    // Last thread does the MPI reduction
    if (args->nbytes > 0) {
      void* remote, *remoteHost = malloc(args->nbytes);
      void* myInitialData = malloc(args->nbytes);
      memcpy(myInitialData, args->expectedHost[0], args->nbytes);
      CUDACHECK(cudaHostRegister(remoteHost, args->nbytes, 0));
      CUDACHECK(cudaHostGetDevicePointer(&remote, remoteHost, 0));
      for (int i=0; i<args->nProcs; i++) {
        if (i == args->proc) {
          MPI_Bcast(myInitialData, args->nbytes, MPI_BYTE, i, MPI_COMM_WORLD);
          free(myInitialData);
        } else {
          MPI_Bcast(remoteHost, args->nbytes, MPI_BYTE, i, MPI_COMM_WORLD);
          Accumulate(args->expected[0], remote, count, type, op);
          cudaDeviceSynchronize();
        }
      }
      CUDACHECK(cudaHostUnregister(remoteHost));
      free(remoteHost);
    }
#endif
    args->sync[0] = 0;
  } else {
    while (args->sync[0]) pthread_yield();
  }
}

void GetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(2*(nranks - 1)))/((double)nranks);
  *busBw = baseBw * factor;
}

void RunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, type, op, comm, stream));
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
