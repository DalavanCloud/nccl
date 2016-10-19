/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef BOOTSTRAP_H_
#define BOOTSTRAP_H_

#include "nccl.h"

struct ncclBootstrap {
  ncclResult_t (*getUniqueId)(ncclUniqueId*);
  struct ncclBootstrap* (*init)(ncclUniqueId*, int, int);
  void (*allGather)(void*, int);
  void (*ringExchange)(void*, int);
};

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out);
struct ncclBootstrap* bootstrapInit(ncclUniqueId* id, int rank, int nranks);
#endif
