/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#include <assert.h>

void print_header() {
  PRINT("# %10s  %12s  %6s  %6s        out-of-place\n", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %6s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type", "root", 
      "time", "algbw", "busbw", "res");
}

void print_line_header (int size, int count, const char *typeName, const char *opName, int root) {
  PRINT("%12i  %12i  %6s  %6i", size, count, typeName, root);
}

void getCollByteCount(size_t *sendbytes, size_t *recvbytes, size_t *procSharedBytes, int *sameExpected, size_t nbytes, int nranks) {
    *sendbytes = nbytes;
    *recvbytes = nbytes;
    *sameExpected = 0;
    *procSharedBytes = nbytes;
}

void InitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  int root_proc = root/(args->nThreads*args->nGpus);
  int root_thread = (root/args->nGpus)%(args->nThreads);
  int root_gpu = root%args->nGpus;

  assert(args->expectedBytes == args->nbytes);

  if (args->thread == 0) args->sync[0] = root_thread;

  if (root_thread == args->thread) { 
      while (args->sync[0] != args->thread) pthread_yield();
     
#ifdef MPI_SUPPORT
      if (root_proc == args->proc) {  
         CUDACHECK(cudaMemcpy(args->procSharedHost,
                    args->sendbuffs[root_gpu],
                    args->nbytes, cudaMemcpyDeviceToHost));
      }
 
      MPI_Bcast(args->procSharedHost, args->nbytes, MPI_BYTE, root_proc, MPI_COMM_WORLD);
#endif

      args->sync[0] = 0;
  }

  while (args->sync[0] != args->thread) pthread_yield();

  for (int i=0; i<args->nGpus; i++) {
     //set expected buf to zero at root, copy over source data at others
     if ((root_proc == args->proc) 
         && (root_thread == args->thread) 
         && (root_gpu == i)) { 
         memset(args->expectedHost[i], 0, args->nbytes); 
     } else { 
         memcpy(args->expectedHost[i], args->procSharedHost, args->nbytes);
     }

     //reset recvbufs to zero
     CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->nbytes));
     CUDACHECK(cudaDeviceSynchronize());
  }

  args->sync[0] = args->thread + 1;

  if (args->thread+1 == args->nThreads) {
    args->sync[0] = 0;
  } else {
    while (args->sync[0]) pthread_yield();
  }
}

void GetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

void RunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int rank; 
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  if (rank == root) { 
      NCCLCHECK(ncclBcast(sendbuff, count, type, root, comm, stream));
  } else { 
      NCCLCHECK(ncclBcast(recvbuff, count, type, root, comm, stream));
  } 
}

void RunTestType(struct threadArgs_t* args, ncclDataType_t type, const char* typeName) {
  for (int i=0; i<args->nProcs*args->nThreads*args->nGpus; i++) { 
     TimeTest(args, type, typeName, (ncclRedOp_t)0, NULL, i, 0);
  }
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
