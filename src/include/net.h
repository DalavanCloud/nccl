/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NET_H_
#define NCCL_NET_H_

#include "nccl.h"

typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

#define EXTCHECK(cmd) do { \
  int err = cmd; \
  if (err != 0) return ncclSystemError; \
} while (false)

static ncclResult_t ncclNetCudaSupport() { EXTCHECK(ncclNet->cudaSupport()); return ncclSuccess; }
static ncclResult_t ncclNetGetHandle(void* handle, void** recvComm) { EXTCHECK(ncclNet->getHandle(handle, recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetConnectHandle(void* handle, void** sendComm) { EXTCHECK(ncclNet->connectHandle(handle, sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetIsend(void* sendComm, void* data, int size, void** request) { EXTCHECK(ncclNet->iSend(sendComm, data, size, request)); return ncclSuccess; }
static ncclResult_t ncclNetIrecv(void* recvComm, void* data, int size, void** request) { EXTCHECK(ncclNet->iRecv(recvComm, data, size, request)); return ncclSuccess; }
static ncclResult_t ncclNetTest(void* request, int* done, int* size) { EXTCHECK(ncclNet->test(request, done, size)); return ncclSuccess; }

static ncclResult_t ncclNetSend(void* sendComm, void* data, int size) {
  void* request;
  EXTCHECK(ncclNetIsend(sendComm, data, size, &request));
  int done = 0;
  while (!done) EXTCHECK(ncclNetTest(request, &done, NULL));
  return ncclSuccess;
}
static ncclResult_t ncclNetRecv(void* recvComm, void* data, int size) {
  void* request;
  EXTCHECK(ncclNetIrecv(recvComm, data, size, &request));
  int done = 0;
  while (!done) EXTCHECK(ncclNetTest(request, &done, NULL));
  return ncclSuccess;
}

#endif
