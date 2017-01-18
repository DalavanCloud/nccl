/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"

int ncclDummyExtTransportEnabled() { return 0; }
int ncclDummyExtTransportCudaSupport() { return 0; }
int ncclDummyExtTransportGetHandle(void* handle, void** recvComm) { return 1; }
int ncclDummyExtTransportConnectHandle(void* handle, void** comm) { return 1; }
int ncclDummyExtTransportISend(void* sendComm, void* data, int size, void** request) { return 1; }
int ncclDummyExtTransportIRecv(void* recvComm, void* data, int size, void** request) { return 1; }
int ncclDummyExtTransportTest(void* request, int* done, int* size) { return 1; }

ncclExtTransport_t ncclDummyExtTransport = {
  ncclDummyExtTransportEnabled,
  ncclDummyExtTransportCudaSupport,
  ncclDummyExtTransportGetHandle,
  ncclDummyExtTransportConnectHandle,
  ncclDummyExtTransportISend,
  ncclDummyExtTransportIRecv,
  ncclDummyExtTransportTest
};

extern "C" __attribute__ ((visibility("default")))
ncclExtTransport_t* ncclExtTransport = &ncclDummyExtTransport;
