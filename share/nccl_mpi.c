/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "mpi.h"
#include "nccl.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

static MPI_Comm ncclMpiComm;

static int numRequests = 0;
MPI_Request* ncclMpiRequests = NULL;
int* ncclMpiRequestUsed = NULL;
pthread_mutex_t ncclMpiRequestsLock = PTHREAD_MUTEX_INITIALIZER;

MPI_Request* ncclMpiGetRequest() {
  pthread_mutex_lock(&ncclMpiRequestsLock);
  for (int i=0; i<numRequests; i++) {
    if (ncclMpiRequestUsed[i] == 0) {
      ncclMpiRequestUsed[i] = 1; 
      pthread_mutex_unlock(&ncclMpiRequestsLock);
      return ncclMpiRequests + i;
    }
  }
  // No free request found, grow the pool
  int newNumRequests = numRequests + 32;
  MPI_Request* newRequests = (MPI_Request*)malloc(newNumRequests*sizeof(MPI_Request));
  int* newUsed = (int*)malloc(newNumRequests*sizeof(int));
  for (int i=0; i<numRequests; i++) {
    newRequests[i] = ncclMpiRequests[i];
    newUsed[i] = ncclMpiRequestUsed[i];
  } 
  for (int i=numRequests; i<newNumRequests; i++)
    newUsed[i] = 0;
  free(ncclMpiRequests);
  ncclMpiRequests = newRequests;
  free(ncclMpiRequestUsed);
  ncclMpiRequestUsed = newUsed;
  numRequests = newNumRequests;
  pthread_mutex_unlock(&ncclMpiRequestsLock);
  return ncclMpiGetRequest();
}

void ncclMpiFreeRequest(MPI_Request* request) {
  pthread_mutex_lock(&ncclMpiRequestsLock);
  ncclMpiRequestUsed[request-ncclMpiRequests] = 0;
  pthread_mutex_unlock(&ncclMpiRequestsLock);
}

int ncclMpiEnabled() {
  int threadLevel;
  MPI_Query_thread(&threadLevel);
  if (threadLevel < MPI_THREAD_MULTIPLE) {
    fprintf(stderr, "Warning : NCCL requires at least MPI_THREAD_SERIALIZED, got %d. MPI bindings are disabled.\n", threadLevel);
    return 0;
  }
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

struct ncclMpiHandle {
  int rank;
  int tag;
};

struct ncclMpiRecvComm {
  int tag;
};

struct ncclMpiSendComm {
  int rank;
  int tag;
};

// Generate a "unique" tag
static int mpiTag = 1;
static void getTag(int *tag) {
  int val = mpiTag;
  while (__sync_val_compare_and_swap(&mpiTag, val, val+1) != val) {
   val++;
  }
  *tag = val;
}

int ncclMpiGetHandle(void* opaqueHandle, void** recvComm) {
  struct ncclMpiRecvComm* comm = (struct ncclMpiRecvComm*)malloc(sizeof(struct ncclMpiRecvComm));
  struct ncclMpiHandle* handle = (struct ncclMpiHandle*) opaqueHandle;
  assert(sizeof(struct ncclMpiHandle) < NCCL_EXT_HANDLE_MAXSIZE);
  int tag;
  getTag(&tag);
  comm->tag = handle->tag = tag;
  int ret = MPI_Comm_rank(ncclMpiComm, &handle->rank);
  *recvComm = comm;
  return ret;
}

int ncclMpiConnectHandle(void* opaqueHandle, void** sendComm) {
  struct ncclMpiSendComm* comm = (struct ncclMpiSendComm*)malloc(sizeof(struct ncclMpiSendComm));
  struct ncclMpiHandle* handle = (struct ncclMpiHandle*) opaqueHandle;
  comm->rank = handle->rank;
  comm->tag = handle->tag;
  *sendComm = comm;
  return 0;
};

int ncclMpiIsend(void* sendComm, void* data, int size, void** request) {
  struct ncclMpiSendComm* comm = (struct ncclMpiSendComm*)sendComm;
  MPI_Request* mpiRequest = ncclMpiGetRequest();
  *request = mpiRequest;
  return MPI_Isend(data, size, MPI_BYTE, comm->rank, comm->tag, ncclMpiComm, mpiRequest);
}

int ncclMpiIrecv(void* recvComm, void* data, int size, void** request) {
  struct ncclMpiRecvComm* comm = (struct ncclMpiRecvComm*)recvComm;
  MPI_Request* mpiRequest = ncclMpiGetRequest();
  *request = mpiRequest;
  return MPI_Irecv(data, size, MPI_BYTE, MPI_ANY_SOURCE, comm->tag, ncclMpiComm, mpiRequest);
}

int ncclMpiTest(void* request, int* done, int* size) {
  MPI_Request* mpiRequest = (MPI_Request*)request;
  MPI_Status status;
  int err = MPI_Test(mpiRequest, done, &status);
  if (*done == 1) {
    if (size) MPI_Get_count(&status, MPI_BYTE, size);
    ncclMpiFreeRequest(request);
  }
  return err;
}

ncclExtTransport_t ncclMpiTransport = {
  ncclMpiEnabled,
  ncclMpiCudaSupport,
  ncclMpiGetHandle,
  ncclMpiConnectHandle,
  ncclMpiIsend,
  ncclMpiIrecv,
  ncclMpiTest
};

void ncclMpiHook(MPI_Comm comm) {
  ncclMpiComm = comm;
  ncclExtTransport = &ncclMpiTransport;
}

