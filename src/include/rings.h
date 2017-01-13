/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_RINGS_H_
#define NCCL_RINGS_H_

ncclResult_t ncclGetRings(int* nrings, int rank, int nranks, int* transports, int* values, int* prev, int* next);

#endif
