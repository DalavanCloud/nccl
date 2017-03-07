/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ibvwrap.h"

#ifndef IBV_DIRECT
#include <dlfcn.h>

static enum { ibvUninitialized, ibvInitializing, ibvInitialized, ibvError } ibvState = ibvUninitialized;

static struct ibv_device** (*ibv_internal_get_device_list)(int *num_devices); 
static struct ibv_context* (ibv_internal_open_device)(struct *ibv_device);
