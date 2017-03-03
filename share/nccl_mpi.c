/*************************************************************************
 *  Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * 
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION, nor the names of their
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ************************************************************************/

#include "mpi.h"
#include "nccl.h"

/************************************************************************
 * This is an example using the NCCL network API to use MPI for
 * inter-node communication.
 *
 * This file should be included as part of the application code.
 ************************************************************************/

/* Functions to be used by the application */

// ncclMpiHook : make NCCL use MPI as inter-node communication system.
// This function should be called after MPI_Init and before any NCCL call
// (in particular ncclCommGetUniqueId and ncclCommInitRank).
// If MPI is used concurrently with NCCL, it is recommended to create a 
// dedicated communicator for NCCL (usually a dup of MPI_COMM_WORLD).
void ncclMpiHook(MPI_Comm comm);

// ncclMpiLock/ncclMpiUnlock : protect MPI calls if MPI is not thread-safe.
// NCCL being an asynchronous communication library, MPI may be called from
// threads. If the MPI implementation is not THREAD_MULTIPLE, it is critical
// to guard other MPI calls in the application using those two functions.
void ncclMpiLock();
void ncclMpiUnlock();

/* NCCL MPI Plugin */

// Functions prototypes
int ncclMpiPtrSupport(int* supportedTypes);
int ncclMpiDevices(int* ndev, int** distances);
int ncclMpiListen(int dev, void* handle, void** listenComm);
int ncclMpiConnectHandle(int dev, void* handle, void** sendComm);
int ncclMpiAccept(void *listenComm, void** recvComm);
int ncclMpiIsend(void* sendComm, void* data, int size, int type, void** request);
int ncclMpiIrecv(void* recvComm, void* data, int size, int type, void** request);
int ncclMpiTest(void* request, int* done, int* size);
int ncclMpiCloseSend(void* sendComm);
int ncclMpiCloseRecv(void* recvComm);
int ncclMpiCloseListen(void* listenComm);

// MPI Net Module
ncclNet_t ncclMpi = {
  "MPI",
  ncclMpiPtrSupport,
  ncclMpiDevices,
  ncclMpiListen,
  ncclMpiConnectHandle,
  ncclMpiAccept,
  ncclMpiIsend,
  ncclMpiIrecv,
  ncclMpiTest,
  ncclMpiCloseSend,
  ncclMpiCloseRecv,
  ncclMpiCloseListen
};

static MPI_Comm ncclMpiComm;

void ncclMpiHook(MPI_Comm comm) {
  ncclMpiComm = comm;
  ncclNet = &ncclMpi;
}

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

pthread_mutex_t ncclMpiGlobalLock = PTHREAD_MUTEX_INITIALIZER;
static int ncclMpiLockMode = -1;

void ncclMpiLock() {
  if (ncclMpiLockMode == 1) pthread_mutex_lock(&ncclMpiGlobalLock);
}
void ncclMpiUnlock() {
  if (ncclMpiLockMode == 1) pthread_mutex_unlock(&ncclMpiGlobalLock);
}

static void ncclMpiGetLockMode() {
  int provided;
  MPI_Query_thread(&provided);
  // MPI implementations may have thread safety bugs; provide a way
  // to force locking.
  char* str = getenv("NCCL_MPI_FORCE_LOCK");
  if (provided < MPI_THREAD_MULTIPLE || (str && atoi(str) == 1)) {
    ncclMpiLockMode = 1;
  } else {
    ncclMpiLockMode = 0;
  }
}

#define MPI_PROTECT(retvar, cmd) do { \
  if (ncclMpiLockMode == -1) ncclMpiGetLockMode(); \
  ncclMpiLock(); \
  retvar = cmd; \
  ncclMpiUnlock(); \
} while (0)

/* Dynamic request pool management */
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

/* Opaque structures */

// We generate a tag for each handle, but our rank doesn't change.
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

static int getCudaSupport() {
  static int cudaSupport = -1;
  if (cudaSupport == -1) {
    char* str = getenv("NCCL_MPI_CUDA_SUPPORT");
    cudaSupport = str ? atoi(str) : 0;
  }
  return cudaSupport;
}

int ncclMpiPtrSupport(int* supportedTypes) {
  *supportedTypes = NCCL_PTR_HOST;
  if (getCudaSupport()) *supportedTypes |= NCCL_PTR_CUDA;
  return 0;
}

int ncclMpiDevices(int* ndev, int** distances) {
  *ndev = 1;
  *distances = (int*)malloc(sizeof(int));
  distances[0][0] = 0;
  return 0;
}

int ncclMpiListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclMpiRecvComm* comm = (struct ncclMpiRecvComm*)malloc(sizeof(struct ncclMpiRecvComm));
  struct ncclMpiHandle* handle = (struct ncclMpiHandle*) opaqueHandle;
  assert(sizeof(struct ncclMpiHandle) < NCCL_NET_HANDLE_MAXSIZE);
  int tag;
  getTag(&tag);
  comm->tag = handle->tag = tag;
  int ret;
  MPI_PROTECT(ret, MPI_Comm_rank(ncclMpiComm, &handle->rank));
  //MPI_PROTECT(ret, MPI_Comm_rank(MPI_COMM_WORLD, &handle->rank));
  *listenComm = comm;
  return ret;
}

// No real need to connect or accept in MPI, just retain the rank and tag.
int ncclMpiConnectHandle(int dev, void* opaqueHandle, void** sendComm) {
  struct ncclMpiSendComm* comm = (struct ncclMpiSendComm*)malloc(sizeof(struct ncclMpiSendComm));
  struct ncclMpiHandle* handle = (struct ncclMpiHandle*) opaqueHandle;
  comm->rank = handle->rank;
  comm->tag = handle->tag;
  *sendComm = comm;
  return 0;
}

int ncclMpiAccept(void *listenComm, void** recvComm) {
  return 0;
}

#define CHECK_PTR(type) do {          \
  if (type == NCCL_PTR_CUDA) {        \
    if (getCudaSupport() == 0)        \
      return 1;                       \
  } else if (type != NCCL_PTR_HOST) { \
    return 1;                         \
  }                                   \
} while(0)

int ncclMpiIsend(void* sendComm, void* data, int size, int type, void** request) {
  printf("ncclMpiIsend\n");
  int ret;
  //CHECK_PTR(type);
  struct ncclMpiSendComm* comm = (struct ncclMpiSendComm*)sendComm;
  MPI_Request* mpiRequest = ncclMpiGetRequest();
  *request = mpiRequest;
  printf("Send : %p %d %d %d %p\n", data, size, comm->rank, comm->tag, mpiRequest);
  MPI_PROTECT(ret, MPI_Isend(data, size, MPI_BYTE, comm->rank, comm->tag, ncclMpiComm, mpiRequest));
  return ret;
}

int ncclMpiIrecv(void* recvComm, void* data, int size, int type, void** request) {
  printf("ncclMpiIrecv\n");
  int ret;
  //CHECK_PTR(type);
  struct ncclMpiRecvComm* comm = (struct ncclMpiRecvComm*)recvComm;
  MPI_Request* mpiRequest = ncclMpiGetRequest();
  *request = mpiRequest;
  //printf("Recv : %p %d %d %p\n", data, size, comm->tag, mpiRequest);
  MPI_PROTECT(ret, MPI_Irecv(data, size, MPI_BYTE, 0/*MPI_ANY_SOURCE*/, 0/*comm->tag*/, ncclMpiComm, mpiRequest));
  return ret;
}

int ncclMpiTest(void* request, int* done, int* size) {
  MPI_Request* mpiRequest = (MPI_Request*)request;
  MPI_Status status;
  int err;
  MPI_PROTECT(err, MPI_Test(mpiRequest, done, &status));
  if (err == 0 && *done == 1) {
    if (size) MPI_PROTECT(err, MPI_Get_count(&status, MPI_BYTE, size));
    ncclMpiFreeRequest(request);
  }
  return err;
}

// No need to close connections in MPI
int ncclMpiCloseSend(void* sendComm) {
  return 0;
}

int ncclMpiCloseRecv(void* recvComm) {
  return 0;
}

int ncclMpiCloseListen(void* listenComm) {
  return 0;
}
