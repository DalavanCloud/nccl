/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
extern "C" {
int __attribute__((weak)) ncclMpiEnabled() { fprintf(stderr, "No MPI\n"); return 0; }
int __attribute__((weak)) ncclMpiCudaSupport() { return 0; }
int __attribute__((weak)) ncclMpiCommRank(int *rank) { return 1; }
int __attribute__((weak)) ncclMpiGetTag(int *tag) { return 1; }
int __attribute__((weak)) ncclMpiIsend(int rank, void* data, int size, int tag, int request) { return 1; }
int __attribute__((weak)) ncclMpiIrecv(int rank, void* data, int size, int tag, int request) { return 1; }
int __attribute__((weak)) ncclMpiTest(int request, int* done) { return 1; }
}

