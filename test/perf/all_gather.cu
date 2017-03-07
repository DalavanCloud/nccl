/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"


void print_header() {
  PRINT("# %10s  %12s  %6s  %6s        out-of-place                    in-place\n", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %7s  %5s  %5s  %7s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type", 
      "time", "algbw", "busbw", "res", "time", "algbw", "busbw", "res");
}

void print_line_header (int size, int count, const char *typeName, const char *opName, int root) {
  PRINT("%12i  %12i  %6s", size, count, typeName);
}

void getCollByteCount(size_t *sendbytes, size_t *recvbytes, size_t *procSharedBytes, int *sameExpected, size_t nbytes, int nranks) {
    *sendbytes = nbytes;
    *recvbytes = nbytes*nranks;
    *sameExpected = 1;
    *procSharedBytes = 0;
}

void InitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t nBytes = args->nbytes;
  size_t count = nBytes / wordSize(type);
  int proc = args->proc;
  int nThreads = args->nThreads;
  int t = args->thread;
  int nGpus = args->nGpus;

  while (args->sync[0] != t) pthread_yield();

  for (int i=0; i<nGpus; i++) {
    int device;
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));

    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];

    CUDACHECK(cudaMemcpy((void *)((uintptr_t)args->expectedHost[0] + ((proc*nThreads + t)*nGpus + i)*nBytes), 
                data, 
                count*wordSize(type), cudaMemcpyDeviceToHost));

    if (in_place == 0) {
      CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }

  args->sync[0] = t + 1;

  if (t+1 == nThreads) {
#ifdef MPI_SUPPORT
    // Last thread does the MPI allgather
    MPI_Allgather(MPI_IN_PLACE, nBytes*nThreads*nGpus, MPI_BYTE, 
        args->expectedHost[0], 
        nBytes*nThreads*nGpus, MPI_BYTE, MPI_COMM_WORLD);
#endif

    args->sync[0] = 0;
  } else {
    while (args->sync[0]) pthread_yield();
  }
}

void GetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = (nranks - 1)/nranks;
  *busBw = baseBw * factor;
}

void RunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclAllGather(sendbuff, recvbuff, count, type, comm, stream));
}

void RunTestType(struct threadArgs_t* args, ncclDataType_t type, const char* typeName) {
  TimeTest(args, type, typeName, (ncclRedOp_t)0, NULL, 0, 1);
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
