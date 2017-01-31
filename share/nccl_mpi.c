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

/************************************************************************
 * This is an example using the NCCL network API to use MPI for inter-node
 * communication.
 *
 * This file should be included in the code and the ncclMpiHook function 
 * should be called before any call to NCCL (ncclCommInitRank in particular).
 *
 ************************************************************************/

#include "mpi.h"
#include "nccl.h"

int ncclMpiGetHandle(void* handle, void** recvComm);
int ncclMpiConnectHandle(void* handle, void** sendComm);
int ncclMpiIsend(void* sendComm, void* data, int size, void** request);
int ncclMpiIrecv(void* recvComm, void* data, int size, void** request);
int ncclMpiTest(void* request, int* done, int* size);
int ncclMpiCloseSend(void* sendComm);
int ncclMpiCloseRecv(void* recvComm);

ncclNet_t ncclMpi = {
  "MPI",
  ncclMpiGetHandle,
  ncclMpiConnectHandle,
  ncclMpiIsend,
  ncclMpiIrecv,
  ncclMpiTest,
  ncclMpiCloseSend,
  ncclMpiCloseRecv
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

static void ncclMpiGetLockMode() {
  int provided;
  MPI_Query_thread(&provided);
  char* str = getenv("NCCL_MPI_FORCE_LOCK");
  if (provided < MPI_THREAD_MULTIPLE || (str && atoi(str) == 1)) {
    ncclMpiLockMode = 1;
  } else {
    ncclMpiLockMode = 0;
  }
}

#define NCCL_MPI_CHECK_LOCK_MODE do { \
  if (ncclMpiLockMode == -1) { \
    ncclMpiGetLockMode(); \
  } \
} while (0)

#define MPI_CALL(retvar, cmd) do { \
  NCCL_MPI_CHECK_LOCK_MODE; \
  if (ncclMpiLockMode == 1) pthread_mutex_lock(&ncclMpiGlobalLock); \
  retvar = cmd; \
  if (ncclMpiLockMode == 1) pthread_mutex_unlock(&ncclMpiGlobalLock); \
} while (0)

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
  assert(sizeof(struct ncclMpiHandle) < NCCL_NET_HANDLE_MAXSIZE);
  int tag;
  getTag(&tag);
  comm->tag = handle->tag = tag;
  int ret;
  MPI_CALL(ret, MPI_Comm_rank(ncclMpiComm, &handle->rank));
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
  int ret;
  MPI_CALL(ret, MPI_Isend(data, size, MPI_BYTE, comm->rank, comm->tag, ncclMpiComm, mpiRequest));
  return ret;
}

int ncclMpiIrecv(void* recvComm, void* data, int size, void** request) {
  struct ncclMpiRecvComm* comm = (struct ncclMpiRecvComm*)recvComm;
  MPI_Request* mpiRequest = ncclMpiGetRequest();
  *request = mpiRequest;
  int ret;
  MPI_CALL(ret, MPI_Irecv(data, size, MPI_BYTE, MPI_ANY_SOURCE, comm->tag, ncclMpiComm, mpiRequest));
  return ret;
}

int ncclMpiTest(void* request, int* done, int* size) {
  MPI_Request* mpiRequest = (MPI_Request*)request;
  MPI_Status status;
  int err;
  MPI_CALL(err, MPI_Test(mpiRequest, done, &status));
  if (err == 0 && *done == 1) {
    if (size) MPI_CALL(err, MPI_Get_count(&status, MPI_BYTE, size));
    ncclMpiFreeRequest(request);
  }
  return err;
}

int ncclMpiCloseSend(void* sendComm) {
  return 0;
}

int ncclMpiCloseRecv(void* recvComm) {
  return 0;
}

