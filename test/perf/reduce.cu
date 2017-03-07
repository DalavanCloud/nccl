/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include <assert.h>
#include "cuda_runtime.h"
#include "common.h"

void print_header() {
  PRINT("# %10s  %12s  %6s  %6s        out-of-place                    in-place\n", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %6s  %6s %7s  %5s  %5s  %7s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type", "op", "root",
      "time", "algbw", "busbw", "res", "time", "algbw", "busbw", "res");
}

void print_line_header (int size, int count, const char *typeName, const char *opName, int root) {
  PRINT("%12i  %12i  %6s  %6s  %6i", size, count, typeName, opName, root);
}

void getCollByteCount(size_t *sendbytes, size_t *recvbytes, size_t *procSharedBytes, int *sameExpected, size_t nbytes, int nranks) {
    *sendbytes = nbytes;
    *recvbytes = nbytes;
    *sameExpected = 0;
    *procSharedBytes = nbytes;
 }

void InitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t count = args->expectedBytes / wordSize(type);
  int root_proc = root/(args->nThreads*args->nGpus);
  int root_gpu = root%args->nGpus;

  assert(args->expectedBytes == args->nbytes);

  while (args->sync[0] != args->thread) pthread_yield();

  for (int i=0; i<args->nGpus; i++) {
    int device;
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];

    if (is_first && i == 0) {
      CUDACHECK(cudaMemcpy(args->procSharedHost, data, count*wordSize(type), cudaMemcpyDeviceToHost));
    } else {
      Accumulate(args->procShared, data, count, type, op);
    }

    if (in_place == 0) {
      CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }

  args->sync[0] = args->thread + 1;

  if (args->thread+1 == args->nThreads) {
#ifdef MPI_SUPPORT
    // Last thread does the MPI reduction
    if (root_proc == args->proc) { 
        void* temp, *tempHost = malloc(args->expectedBytes);
        CUDACHECK(cudaHostRegister(tempHost, args->expectedBytes, 0));
        CUDACHECK(cudaHostGetDevicePointer(&temp, tempHost, 0));

        for (int i=0; i<args->nProcs; i++) {
            if (i == args->proc) continue;
            MPI_Recv(tempHost, args->expectedBytes, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            Accumulate(args->procShared, temp, count, type, op);
        }
        CUDACHECK(cudaDeviceSynchronize());

        CUDACHECK(cudaHostUnregister(tempHost));
        free(tempHost);
    } else {
        MPI_Send(args->procSharedHost, args->expectedBytes, MPI_BYTE, root_proc, 0, MPI_COMM_WORLD);
    }
#endif
    args->sync[0] = 0;
  } else {
    while (args->sync[0]) pthread_yield();
  }

  //if root fill expected bytes with reduced data
  // else if in_place, leave fill it with original data, else set to zero
  for (int i=0; i<args->nGpus; i++) {
      int rank = (args->proc*args->nThreads + args->thread)*args->nGpus + i;
      if (rank == root) { 
          memcpy(args->expectedHost[root_gpu], args->procSharedHost, args->expectedBytes); 
      } else { 
         if (in_place == 1) {
              CUDACHECK(cudaMemcpy(args->expectedHost[i], args->recvbuffs[i], args->expectedBytes, cudaMemcpyDeviceToHost));
          } else {
              memset(args->expectedHost[i], 0, args->expectedBytes); 
          }
      } 
  }
}

void GetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(2*(nranks - 1)))/((double)nranks);
  *busBw = baseBw * factor;
}

void RunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, type, op, root, comm, stream));
}

void RunTestOp(struct threadArgs_t* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  for (int i=0; i<args->nProcs*args->nThreads*args->nGpus; i++) {
      TimeTest(args, type, typeName, op, opName, i, 1);
  }
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
