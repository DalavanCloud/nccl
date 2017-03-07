/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ibvwrap.h"

#ifndef IBV_DIRECT
#include <dlfcn.h>
#include "core.h"

static enum { ibvUninitialized, ibvInitializing, ibvInitialized, ibvError } ibvState = ibvUninitialized;

static struct ibv_device** (*ibv_internal_get_device_list)(int *num_devices); 
static void (*ibv_internal_free_device_list)(struct ibv_device **list);
static struct ibv_context* (*ibv_internal_open_device)(struct ibv_device* device);
static int (*ibv_internal_poll_cq)(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc);
static int (*ibv_internal_post_send)(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr);
static int (*ibv_post_recv)(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr);


inline int wrap_ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
  if (ibv_internal_poll_cq == NULL) {
     WARN("lib wrapper not initialized.");
     return ncclLibWrapperNotSet;
  }
  int ret = ibv_internal_poll_cq(cq, num_entries, wc);
  if (!ret) {
    WARN("ibv_poll_cq() failed");
    return ncclSystemError; 
  }
  return ncclSuccess;
}

#endif
