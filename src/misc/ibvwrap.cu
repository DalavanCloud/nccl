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

ncclResult_t wrap_ibv_symbols(void) {
  if (ibvState == ibvInitialized)
    return ncclSuccess;
  if (ibvState == ibvError)
    return ncclSystemError;
  
  if (__sync_bool_compare_and_swap(&ibvState, ibvUninitialized, ibvInitializing) == false) {
    // Another thread raced in front of us. Wait for it to be done.
    while (ibvState == ibvInitializing) pthread_yield();
    return (ibvState == ibvInitialized) ? ncclSuccess : ncclSystemError;
  }

  static void* ibvhandle = NULL;
  void* tmp;
  void** cast;

  ibvhandle=dlopen("libibverbs.so", RTLD_NOW);
  if (!ibvhandle) {
    ibvhandle=dlopen("libibverbs.so.1", RTLD_NOW);
    if (!ibvhandle) {
      WARN("Failed to open libibverbs.so[.1]");
      goto teardown;
    }
  }

  #define LOAD_SYM(handle, symbol, funcptr) do {         \
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                         \
    if (tmp == NULL) {                                   \ 
      WARN("dlsym failed on %s - %s", symbol, dlerror());\
      goto teardown;                                     \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

  LOAD_SYM(ibvhandle, "ibv_get_device_list", ibv_internal_get_device_list);

  ibvState = ibvInitialized;
  return ncclSuccess;

  teardown:
  ibv_internal_get_device_list = NULL;

  if (ibvhandle != NULL) dlclose(ibvhandle);
  ibvState = ibvError;
  return ncclSystemError;
}

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
