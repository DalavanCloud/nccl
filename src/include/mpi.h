/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#define MPICHECK(cmd) do { \
  int err = cmd; \
  if (err != 0) { \
    WARN("MPI Failure %d\n", err); \
    return ncclSystemError; \
  } \
} while (false)

extern "C" {
int ncclMpiEnabled();
int ncclMpiCudaSupport();

int ncclMpiCommRank(int *rank);
int ncclMpiGetTag(int *tag);
int ncclMpiSend(int rank, void* data, int size, int tag);
int ncclMpiRecv(int rank, void* data, int size, int tag);
}

