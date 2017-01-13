/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"

struct ncclBootstrap {
  ncclResult_t (*getUniqueId)(ncclUniqueId*);
  ncclResult_t (*init)(ncclUniqueId*, int, int, void**);
  ncclResult_t (*allGather)(void*, void*, int);
  ncclResult_t (*ringExchange)(void*, void*, int, int, int);
};

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out);
ncclResult_t bootstrapInit(ncclUniqueId* id, int rank, int nranks, struct ncclBootstrap** bootstrap, void** commState);
#endif
