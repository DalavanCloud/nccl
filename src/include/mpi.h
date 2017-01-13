/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_MPI_H_
#define NCCL_MPI_H_

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
int ncclMpiIsend(int rank, void* data, int size, int tag, int request);
int ncclMpiIrecv(int rank, void* data, int size, int tag, int request);
int ncclMpiTest(int request, int* done, int* size);

#define MPICHECKINTERNAL(cmd) do { \
  int err = cmd; \
  if (err != 0) { \
    WARN("MPI Failure %d\n", err); \
    return err; \
  } \
} while (false)

static int ncclMpiSend(int rank, void* data, int size, int tag) {
  MPICHECKINTERNAL(ncclMpiIsend(rank, data, size, tag, -1));
  int done = 0;
  while (done == 0)
    MPICHECKINTERNAL(ncclMpiTest(-1, &done, NULL));
  return 0;
}

static int ncclMpiRecv(int rank, void* data, int size, int tag) {
  MPICHECKINTERNAL(ncclMpiIrecv(rank, data, size, tag, -1));
  int done = 0;
  while (done == 0)
    MPICHECKINTERNAL(ncclMpiTest(-1, &done, NULL));
  return 0;
}

}
#endif
