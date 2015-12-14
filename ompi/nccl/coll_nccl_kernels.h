/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2015      NVIDIA Corporation. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef COLL_NCCL_KERNELS_H
#define COLL_NCCL_KERNELS_H

#include <nccl.h>
#include "ompi/datatype/ompi_datatype.h"
#ifdef __cplusplus
extern "C"
#endif
void
mca_coll_nccl_cuda_op_reduce(void * a, void * b, int N,
                        struct ompi_datatype_t *dtype,
                        ncclRedOp_t op, cudaStream_t stream);
#endif
