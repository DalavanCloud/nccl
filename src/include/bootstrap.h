/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out);
ncclResult_t bootstrapInit(ncclUniqueId* id, int rank, int nranks, void** commState);
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapRingExchange(void* commState, void* prevNextData, int prev, int next, int size);
#endif
