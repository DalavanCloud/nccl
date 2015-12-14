/*
 * Copyright (c) 2015      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef COLL_NCCL_MULTICOLL_H
#define COLL_NCCL_MULTICOLL_H

#include <nccl.h>
#include <ompi/datatype/ompi_datatype.h>
#include "coll_nccl.h"


#define COLL_NCCL_MULTI_COMPUTE_BLOCKCOUNT( COUNT, NUM_BLOCKS, SPLIT_INDEX,       \
                                       EARLY_BLOCK_COUNT, LATE_BLOCK_COUNT ) \
    EARLY_BLOCK_COUNT = LATE_BLOCK_COUNT = COUNT / NUM_BLOCKS;               \
    SPLIT_INDEX = COUNT % NUM_BLOCKS;                                        \
    if (0 != SPLIT_INDEX) {                                                  \
        EARLY_BLOCK_COUNT = EARLY_BLOCK_COUNT + 1;                           \
    }


#define COLL_NCCL_MULTI_COMPUTED_SEGCOUNT(SEGSIZE, TYPELNG, SEGCOUNT)        \
    if( ((SEGSIZE) >= (TYPELNG)) &&                                     \
        ((SEGSIZE) < ((TYPELNG) * (SEGCOUNT))) ) {                      \
        size_t residual;                                                \
        (SEGCOUNT) = (int)((SEGSIZE) / (TYPELNG));                      \
        residual = (SEGSIZE) - (SEGCOUNT) * (TYPELNG);                  \
        if( residual > ((TYPELNG) >> 1) )                               \
            (SEGCOUNT)++;                                               \
    }


int
mca_coll_nccl_multi_allreduce(void *sbuf, void *rbuf, int count,
                              struct ompi_datatype_t *dtype,
                              ncclRedOp_t op,
                              struct ompi_communicator_t *comm,
                              struct mca_coll_nccl_module_t *module);

int
mca_coll_nccl_multi_reduce(void *sbuf, void *rbuf, int count,
                           struct ompi_datatype_t *dtype,
                           ncclRedOp_t op,
                           int root,
                           struct ompi_communicator_t *comm,
                           struct mca_coll_nccl_module_t *module);

int
mca_coll_nccl_multi_bcast(void * buf, int count,
                          struct ompi_datatype_t *dtype,
                          int root,
                          struct ompi_communicator_t *comm,
                          struct mca_coll_nccl_module_t *module);
#endif
