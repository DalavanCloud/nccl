/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "mpi.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// Export this symbol if the application wants to set it to a NCCL-specific communicator
MPI_Comm ncclMpiComm = MPI_COMM_WORLD;

static int numRequests = 0;
MPI_Request* ncclMpiRequests = NULL;

#define CHECKREQINDEX(i) do { \
  if (i+1 < 0) return 1; \
  /* Prevent random index bugs, reqs should be used in increasing order */ \
  if (i+1 > numRequests + 32) return 1; \
  if (i+1 >= numRequests) { \
    numRequests += 32; \
    ncclMpiRequests = realloc(ncclMpiRequests, numRequests*sizeof(MPI_Request)); \
    int r; \
    for (r=numRequests-32; r<numRequests; r++) ncclMpiRequests[r] = MPI_REQUEST_NULL; \
  } \
} while (0);

int ncclMpiEnabled() {
  int threadLevel;
  MPI_Query_thread(&threadLevel);
  if (threadLevel < MPI_THREAD_MULTIPLE) {
    fprintf(stderr, "Warning : NCCL requires at least MPI_THREAD_SERIALIZED, got %d. MPI bindings are disabled.\n", threadLevel);
    return 0;
  }
  // Initialize some requests
  CHECKREQINDEX(-1);
  return 1;
}

int ncclMpiCudaSupport() {
  static int mpiCudaSupport = -1;
  if (mpiCudaSupport == -1) {
    char* str = getenv("NCCL_MPI_GDRDMA");
    mpiCudaSupport = str == NULL ? 0 : atoi(str);
  }
  return mpiCudaSupport;
}

int ncclMpiCommRank(int *rank) {
  return MPI_Comm_rank(ncclMpiComm, rank);
}

// Generate a "unique" tag
static int ncclMpiTag = 1;
int ncclMpiGetTag(int *tag) {
  int val = ncclMpiTag;
  while (__sync_val_compare_and_swap(&ncclMpiTag, val, val+1) != val) {
   val++;
  }
  *tag = val;
  return 0;
}

int ncclMpiIsend(int rank, void* data, int size, int tag, int request) {
  CHECKREQINDEX(request);
  return MPI_Isend(data, size, MPI_BYTE, rank, tag, ncclMpiComm, ncclMpiRequests+request+1);
}

int ncclMpiIrecv(int rank, void* data, int size, int tag, int request) {
  CHECKREQINDEX(request);
  return MPI_Irecv(data, size, MPI_BYTE, rank == -1 ? MPI_ANY_SOURCE : rank, tag, ncclMpiComm, ncclMpiRequests+request+1);
}

int ncclMpiTest(int request, int* done, int* size) {
  CHECKREQINDEX(request);
  MPI_Status status;
  int err = MPI_Test(ncclMpiRequests+request+1, done, &status);
  if (size) MPI_Get_count(&status, MPI_BYTE, size);
  return err;
}
