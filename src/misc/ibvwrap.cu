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
const char * (*ibv_internal_get_device_name)(struct ibv_device *device);
static struct ibv_context* (*ibv_internal_open_device)(struct ibv_device* device);
static int (*ibv_internal_close_device)(struct ibv_context *context);
static int (*ibv_internal_get_async_event)(struct ibv_context *context, struct ibv_async_event *event);
static void (*ibv_internal_ack_async_event)(struct ibv_async_event *event);
static int (*ibv_internal_query_device)(struct ibv_context *context, struct ibv_device_attr *device_attr);
static int (*ibv_internal_query_port)(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr);
static struct ibv_pd * (*ibv_internal_alloc_pd)(struct ibv_context *context);
static struct ibv_mr * (*ibv_internal_reg_mr)(struct ibv_pd *pd, void *addr, size_t length, int access);
static int (*ibv_internal_dereg_mr)(struct ibv_mr *mr);
static struct ibv_comp_channel * (*ibv_internal_create_comp_channel)(struct ibv_context *context);
static struct ibv_cq * (*ibv_internal_create_cq)(struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector);
static int (*ibv_internal_poll_cq)(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc);
static struct ibv_qp * (*ibv_internal_create_qp)(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
static int (*ibv_internal_modify_qp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
static int (*ibv_internal_post_send)(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr);
static int (*ibv_internal_post_recv)(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr);
static const char * (*ibv_internal_event_type_str)(enum ibv_event_type event);

ncclResult_t wrap_ibv_symbols(void) {
  INFO("wrap_ibv_symbols");
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
  LOAD_SYM(ibvhandle, "ibv_free_device_list", ibv_internal_free_device_list);
  LOAD_SYM(ibvhandle, "ibv_get_device_name", ibv_internal_get_device_name);
  LOAD_SYM(ibvhandle, "ibv_opne_device", ibv_internal_open_device);
  LOAD_SYM(ibvhandle, "ibv_close_device", ibv_internal_close_device);
  LOAD_SYM(ibvhandle, "ibv_get_async_event", ibv_internal_get_async_event);
  LOAD_SYM(ibvhandle, "ibv_ack_async_event", ibv_internal_ack_async_event);
  LOAD_SYM(ibvhandle, "ibv_query_device", ibv_internal_query_device);
  LOAD_SYM(ibvhandle, "ibv_query_port", ibv_internal_query_port);
  LOAD_SYM(ibvhandle, "ibv_alloc_pd", ibv_internal_alloc_pd);
  LOAD_SYM(ibvhandle, "ibv_reg_mr", ibv_internal_reg_mr);
  LOAD_SYM(ibvhandle, "ibv_dereg_mr", ibv_internal_dereg_mr);
  LOAD_SYM(ibvhandle, "ibv_create_comp_channel", ibv_internal_create_comp_channel);
  LOAD_SYM(ibvhandle, "ibv_create_cq", ibv_internal_create_cq);
  LOAD_SYM(ibvhandle, "ibv_poll_cq", ibv_internal_poll_cq);
  LOAD_SYM(ibvhandle, "ibv_create_qp", ibv_internal_create_qp);
  LOAD_SYM(ibvhandle, "ibv_modify_qp", ibv_internal_modify_qp);
  LOAD_SYM(ibvhandle, "ibv_post_send", ibv_internal_post_send);
  LOAD_SYM(ibvhandle, "ibv_post_recv", ibv_internal_post_recv);
  LOAD_SYM(ibvhandle, "ibv_event_type_str", ibv_internal_event_type_str);

  ibvState = ibvInitialized;
  return ncclSuccess;

  teardown:
  ibv_internal_get_device_list = NULL;
  ibv_internal_free_device_list = NULL;
  ibv_internal_get_device_name = NULL;
  ibv_internal_open_device = NULL;
  ibv_internal_close_device = NULL;
  ibv_internal_get_async_event = NULL;
  ibv_internal_ack_async_event = NULL;
  ibv_internal_query_device = NULL;
  ibv_internal_query_port = NULL;
  ibv_internal_alloc_pd = NULL;
  ibv_internal_reg_mr = NULL;
  ibv_internal_dereg_mr = NULL;
  ibv_internal_create_comp_channel = NULL;
  ibv_internal_create_cq = NULL;
  ibv_internal_create_qp = NULL;
  ibv_internal_modify_qp = NULL;
  ibv_internal_post_send = NULL;
  ibv_internal_post_recv = NULL;
  ibv_internal_event_type_str = NULL;

  if (ibvhandle != NULL) dlclose(ibvhandle);
  ibvState = ibvError;
  return ncclSystemError;
}

struct ibv_device **wrap_ibv_get_device_list(int *num_devices) {
  if (ibv_internal_get_device_list == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  return ibv_internal_get_device_list(num_devices);
}

void wrap_ibv_free_device_list(struct ibv_device **list) {
  if (ibv_internal_free_device_list == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  ibv_internal_free_device_list(list);
}

const char *wrap_ibv_get_device_name(struct ibv_device *device) {
  if (ibv_internal_get_device_name == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  return ibv_internal_get_device_name(device);
}

struct ibv_context *wrap_ibv_open_device(struct ibv_device *device) { /*returns 0 on success, -1 on failure*/
  if (ibv_internal_open_device == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  return ibv_internal_open_device(device);
}

int wrap_ibv_close_device(struct ibv_context *context) { /*returns 0 on success, -1 on failure*/
  if (ibv_internal_close_device == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  int ret = ibv_internal_close_device(context);
  if (ret != IBV_SUCCESS) {
    WARN("ibv_close_device() failed");
    return ncclSystemError; 
  }
  return ncclSuccess;
}

int wrap_ibv_get_async_event(struct ibv_context *context, struct ibv_async_event *event) {
  if (ibv_internal_get_async_event == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  int ret = ibv_internal_get_async_event(context, event);
  if (ret != IBV_SUCCESS) {
    WARN("ibv_get_async_event() failed");
    return ncclSystemError; 
  }
  return ncclSuccess;
}

void wrap_ibv_ack_async_event(struct ibv_async_event *event) {
  if (ibv_internal_ack_async_event == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  ibv_internal_ack_async_event(event);
}

int wrap_ibv_query_device(struct ibv_context *context, struct ibv_device_attr *device_attr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  if (ibv_internal_query_device == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  int ret = ibv_internal_query_device(context, device_attr);
  if (ret != IBV_SUCCESS) {
    WARN("ibv_query_device() failed");
    return ncclSystemError; 
  }
  return ncclSuccess;
}

int wrap_ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  if (ibv_internal_query_port == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  int ret = ibv_internal_query_port(context, port_num, port_attr);
  if (ret != IBV_SUCCESS) {
    WARN("ibv_query_port() failed");
    return ncclSystemError; 
  }
  return ncclSuccess;
}

struct ibv_pd *wrap_ibv_alloc_pd(struct ibv_context *context) {
  if (ibv_internal_alloc_pd == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  return ibv_internal_alloc_pd(context);
}

struct ibv_mr *wrap_ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access) {
  if (ibv_internal_reg_mr == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  return ibv_internal_reg_mr(pd, addr, length, access);
}

int wrap_ibv_dereg_mr(struct ibv_mr *mr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  if (ibv_internal_dereg_mr == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  int ret = ibv_internal_dereg_mr(mr);
  if (ret != IBV_SUCCESS) {
    WARN("ibv_dereg_mr() failed");
    return ncclSystemError; 
  }
  return ncclSuccess;
}

struct ibv_comp_channel *wrap_ibv_create_comp_channel(struct ibv_context *context) {
  if (ibv_internal_create_comp_channel == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  return ibv_internal_create_comp_channel(context);
}

struct ibv_cq *wrap_ibv_create_cq(struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector) {
  if (ibv_internal_create_cq == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  return ibv_internal_create_cq(context, cqe, cq_context, channel, comp_vector);
}

int wrap_ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
  if (ibv_internal_poll_cq == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  int ret = ibv_internal_poll_cq(cq, num_entries, wc);
  if (ret != IBV_SUCCESS) {
    WARN("ibv_poll_cq() failed");
    return ncclSystemError; 
  }
  return ncclSuccess;
}

struct ibv_qp *wrap_ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr) {
  if (ibv_internal_create_qp == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  return ibv_internal_create_qp(pd, qp_init_attr);
}

int wrap_ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  if (ibv_internal_modify_qp == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  int ret = ibv_internal_modify_qp(qp, attr, attr_mask);
  if (ret != IBV_SUCCESS) {
    WARN("ibv_modify_qp() failed");
    return ncclSystemError; 
  }
  return ncclSuccess;
}

int wrap_ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  printf("wrap_ibv_post_send");
  if (ibv_internal_post_send == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  int ret = ibv_internal_post_send(qp, wr, bad_wr);
  if (ret != IBV_SUCCESS) {
    WARN("ibv_post_send() failed");
    return ncclSystemError; 
  }
  return ncclSuccess;
}

int wrap_ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  printf("wrap_ibv_post_recv");
  if (ibv_internal_post_recv == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  int ret = ibv_internal_post_recv(qp, wr, bad_wr);
  if (ret != IBV_SUCCESS) {
    WARN("ibv_post_recv() failed");
    return ncclSystemError; 
  }
  return ncclSuccess;
}

const char *wrap_ibv_event_type_str(enum ibv_event_type event) {
  if (ibv_internal_event_type_str == NULL) {
     WARN("lib wrapper not initialized.");
     exit(-1);
  }
  return ibv_internal_event_type_str(event);
}

#endif
