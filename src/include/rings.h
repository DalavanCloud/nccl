/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_RINGS_H_
#define NCCL_RINGS_H_

/* Get the default number of threads based on the GPU generation */
/* https://developer.nvidia.com/cuda-gpus
 * 5 = Maxwell, 6 = Pascal, 7 = Volta
 */
static int getDefaultThreads() {
  return ncclCudaCompCap() >= 5 ? 256 : 512;
}

static size_t getRingThreshold(int rank, int minCompCap) {
  size_t threshold = DEFAULT_SINGLE_RING_THRESHOLD;
  if (minCompCap == 7)
    // Double the default threshold on Volta
    threshold <<= 1;

  char* str = getenv("NCCL_SINGLE_RING_THRESHOLD");
  if (str != NULL) {
    errno = 0;
    threshold = strtol(str, NULL, 0);
    if (errno == ERANGE) {
      INFO("invalid NCCL_SINGLE_RING_THRESHOLD: %s, using default %lu",
           str, DEFAULT_SINGLE_RING_THRESHOLD);
      threshold = DEFAULT_SINGLE_RING_THRESHOLD;
    }
  }

  if (rank == 0) INFO("NCCL_SINGLE_RING_THRESHOLD=%ld", threshold);

  return threshold;
}

ncclResult_t ncclGetRings(int* nrings, int* nthreads, int rank, int nranks, int* transports, int* values, int* prev, int* next);

#endif
