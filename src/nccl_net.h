/*************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NET_H_
#define NCCL_NET_H_

#define NCCL_NET_MAJOR 1
#define NCCL_NET_MINOR 0

#ifdef __cplusplus
extern "C" {
#endif

// Inter-node transport to replace the internal transports.
// An example using MPI can be found in share/nccl_mpi.c.
#define NCCL_NET_HANDLE_MAXSIZE 64

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2

#define NCCL_MAX_SCORE 0x7

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Return the number of network devices with their scores relative to the 
  // current CUDA device. This call should allocate the 'scores' array which
  // will be freed by NCCL.
  int (*devices)(int* ndev, int** scores);
  // Return whether this device supports host pointers and/or CUDA pointers
  // as data from the current GPU. Supported types should be composed with 
  // NCCL_PTR_HOST and NCCL_PTR_CUDA.
  int (*ptrSupport)(int dev, int* supportedTypes);
  // Create a receiving object and provide a handle to connect to it. The 
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged 
  // between ranks to create a connection.
  int (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  int (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connectHandle
  int (*accept)(void* listenComm, void** recvComm);
  // Asynchronous send to a peer. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  int (*isend)(void* sendComm, void* data, int size, int type, void** request);
  // Asynchronous recv from a peer. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  int (*irecv)(void* recvComm, void* data, int size, int type, void** request);
  // Test whether a request is complete and return the size received (can be less than requested).
  int (*test)(void* request, int* done, int* size);
  // Close and free send/recv comm objects
  int (*closeSend)(void* sendComm);
  int (*closeRecv)(void* recvComm);
  int (*closeListen)(void* listenComm);
} ncclNet_t;

extern ncclNet_t* ncclNet;

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard

