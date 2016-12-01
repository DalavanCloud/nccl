/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "mpi.h"

#include "mpi-ext.h" // Needed for CUDA aware detection

#include <pthread.h>
#include <stdio.h>

static int ncclMpiThreadLevel;
static pthread_mutex_t ncclMpiGlobalLock = PTHREAD_MUTEX_INITIALIZER;
// Export this symbol if the application wants to set it to a NCCL-specific communicator
MPI_Comm ncclMpiComm = MPI_COMM_WORLD;

#define MPI_LOCK \
  if (ncclMpiThreadLevel < MPI_THREAD_MULTIPLE) pthread_mutex_lock(&ncclMpiGlobalLock);

#define MPI_UNLOCK \
  if (ncclMpiThreadLevel < MPI_THREAD_MULTIPLE) pthread_mutex_unlock(&ncclMpiGlobalLock);

int ncclMpiEnabled() {
  MPI_Query_thread(&ncclMpiThreadLevel);
  if (ncclMpiThreadLevel < MPI_THREAD_SERIALIZED) {
    fprintf(stderr, "Warning : NCCL requires at least MPI_THREAD_SERIALIZED, got %d. MPI bindings are disabled.\n", ncclMpiThreadLevel);
    return 0;
  }
  return 1;
}

int ncclMpiCudaSupport() {
//#if defined(MPIX_CUDA_AWARE_SUPPORT)
//  return MPIX_Query_cuda_support();
//#else
  return 0;
//#endif
}

int ncclMpiCommRank(int *rank) {
  MPI_LOCK;
  int err = MPI_Comm_rank(ncclMpiComm, rank);
  MPI_UNLOCK;
  return err;
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

int ncclMpiSend(int rank, void* data, int size, int tag) {
  MPI_Request request;
  int done = 0;
  MPI_LOCK;
  int err = MPI_Isend(data, size, MPI_BYTE, rank, tag, ncclMpiComm, &request);
  if (err != MPI_SUCCESS) return err;
  MPI_UNLOCK;
  while (!done) {
    MPI_LOCK;
    err = MPI_Test(&request, &done, MPI_STATUS_IGNORE); 
    if (err != MPI_SUCCESS) return err;
    MPI_UNLOCK;
  }
  return 0;
}

int ncclMpiRecv(int rank, void* data, int size, int tag) {
  int source = rank == -1 ? MPI_ANY_SOURCE : rank;
  MPI_Request request;
  int done = 0;
  MPI_LOCK;
  int err = MPI_Irecv(data, size, MPI_BYTE, rank == -1 ? MPI_ANY_SOURCE : rank, tag, MPI_COMM_WORLD, &request);
  if (err != MPI_SUCCESS) return err;
  MPI_UNLOCK;
  while (!done) {
    MPI_LOCK;
    err = MPI_Test(&request, &done, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS) return err;
    MPI_UNLOCK;
  }
  return 0;
}
