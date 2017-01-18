/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_EXT_H_
#define NCCL_EXT_H_

#include "nccl.h"

typedef char ncclExtHandle_t[NCCL_EXT_HANDLE_MAXSIZE];

#define EXTCHECK(cmd) do { \
  int err = cmd; \
  if (err != 0) return ncclSystemError; \
} while (false)

static ncclResult_t ncclExtEnabled() { return ncclExtTransport->enabled() ? ncclSuccess : ncclInternalError; }
static ncclResult_t ncclExtCudaSupport() { EXTCHECK(ncclExtTransport->cudaSupport()); return ncclSuccess; }
static ncclResult_t ncclExtGetHandle(void* handle, void** recvComm) { EXTCHECK(ncclExtTransport->getHandle(handle, recvComm)); return ncclSuccess; }
static ncclResult_t ncclExtConnectHandle(void* handle, void** sendComm) { EXTCHECK(ncclExtTransport->connectHandle(handle, sendComm)); return ncclSuccess; }
static ncclResult_t ncclExtIsend(void* sendComm, void* data, int size, void** request) { EXTCHECK(ncclExtTransport->iSend(sendComm, data, size, request)); return ncclSuccess; }
static ncclResult_t ncclExtIrecv(void* recvComm, void* data, int size, void** request) { EXTCHECK(ncclExtTransport->iRecv(recvComm, data, size, request)); return ncclSuccess; }
static ncclResult_t ncclExtTest(void* request, int* done, int* size) { EXTCHECK(ncclExtTransport->test(request, done, size)); return ncclSuccess; }

static ncclResult_t ncclExtSend(void* sendComm, void* data, int size) {
  void* request;
  EXTCHECK(ncclExtIsend(sendComm, data, size, &request));
  int done = 0;
  while (!done) EXTCHECK(ncclExtTest(request, &done, NULL));
  return ncclSuccess;
}
static ncclResult_t ncclExtRecv(void* recvComm, void* data, int size) {
  void* request;
  EXTCHECK(ncclExtIrecv(recvComm, data, size, &request));
  int done = 0;
  while (!done) EXTCHECK(ncclExtTest(request, &done, NULL));
  return ncclSuccess;
}

#endif
