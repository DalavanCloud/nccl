/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef BOOTSTRAP_SOCKET_H_
#define BOOTSTRAP_SOCKET_H_

#include "bootstrap.h"

ncclResult_t bootstrapSocketGetUniqueId(ncclUniqueId* out);
struct ncclBootstrap* bootstrapSocketInit(ncclUniqueId* commId, int rank, int nranks);
void bootstrapSocketAllGather(void* allData, int size);
void bootstrapSocketRingExchange(void* prevNextData, int size);

struct ncclBootstrap bootstrapSocket = {
  bootstrapSocketGetUniqueId,
  bootstrapSocketInit,
  bootstrapSocketAllGather,
  bootstrapSocketRingExchange
};
#endif
