/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_RINGS_H_
#define NCCL_RINGS_H_

/* Get the default number of threads based on the GPU generation */
static int getDefaultThreads() {
  return ncclCudaCompCap() >= 5 ? 256 : 512;
}

ncclResult_t ncclGetRings(int* nrings, int* nthreads, int rank, int nranks, int* transports, int* values, int* prev, int* next);

#endif
