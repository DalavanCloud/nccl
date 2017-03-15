/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include "nccl.h"
#include "nccl_net.h"

typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

#define NETCHECK(cmd) do { \
  int err = cmd; \
  if (err != 0) { \
    WARN("Net transport returned error code %d", err); \
    return ncclSystemError; \
  } \
} while (false)

// Translation to external API
static const char* ncclNetName() { return ncclNet->name; }
static ncclResult_t ncclNetDevices(int* ndev, int** distances) { NETCHECK(ncclNet->devices(ndev, distances)); return ncclSuccess; }
static ncclResult_t ncclNetPtrSupport(int dev, int* supportedTypes) { NETCHECK(ncclNet->ptrSupport(dev, supportedTypes)); return ncclSuccess; }
static ncclResult_t ncclNetListen(int dev, void* handle, void** listenComm) { NETCHECK(ncclNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t ncclNetConnect(int dev, void* handle, void** sendComm) { NETCHECK(ncclNet->connect(dev, handle, sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetAccept(void* listenComm, void** recvComm) { NETCHECK(ncclNet->accept(listenComm, recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetIsend(void* sendComm, void* data, int size, int type, void** request) { NETCHECK(ncclNet->isend(sendComm, data, size, type, request)); return ncclSuccess; }
static ncclResult_t ncclNetIrecv(void* recvComm, void* data, int size, int type, void** request) { NETCHECK(ncclNet->irecv(recvComm, data, size, type, request)); return ncclSuccess; }
static ncclResult_t ncclNetTest(void* request, int* done, int* size) { NETCHECK(ncclNet->test(request, done, size)); return ncclSuccess; }

// Additional sync functions based on async + test for bootstrap, using host ptrs.
static ncclResult_t ncclNetSend(void* sendComm, void* data, int size) {
  void* request;
  NETCHECK(ncclNetIsend(sendComm, data, size, NCCL_PTR_HOST, &request));
  int done = 0;
  while (!done) NETCHECK(ncclNetTest(request, &done, NULL));
  return ncclSuccess;
}
static ncclResult_t ncclNetRecv(void* recvComm, void* data, int size) {
  void* request;
  NETCHECK(ncclNetIrecv(recvComm, data, size, NCCL_PTR_HOST, &request));
  int done = 0;
  while (!done) NETCHECK(ncclNetTest(request, &done, NULL));
  return ncclSuccess;
}

static ncclResult_t ncclNetCloseSend(void* sendComm) { NETCHECK(ncclNet->closeSend(sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseRecv(void* recvComm) { NETCHECK(ncclNet->closeRecv(recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseListen(void* listenComm) { NETCHECK(ncclNet->closeListen(listenComm)); return ncclSuccess; }

extern bool ncclIbSupport();
extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

#endif
