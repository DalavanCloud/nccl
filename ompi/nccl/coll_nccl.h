/*
 * Copyright (c) 2014      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2014-2015 NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_NCCL_EXPORT_H
#define MCA_COLL_NCCL_EXPORT_H

#include "ompi_config.h"

#include "mpi.h"

#include "opal/class/opal_object.h"
#include "opal/mca/mca.h"

#include "ompi/constants.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/communicator/communicator.h"

#include <cuda_runtime.h>
#include <nccl.h>

BEGIN_C_DECLS

/* API functions */

int mca_coll_nccl_init_query(bool enable_progress_threads,
                             bool enable_mpi_threads);
mca_coll_base_module_t
*mca_coll_nccl_comm_query(struct ompi_communicator_t *comm,
                          int *priority);

int mca_coll_nccl_module_enable(mca_coll_base_module_t *module,
                                struct ompi_communicator_t *comm);

int
mca_coll_nccl_allreduce(void *sbuf, void *rbuf, int count,
                        struct ompi_datatype_t *dtype,
                        struct ompi_op_t *op,
                        struct ompi_communicator_t *comm,
                        mca_coll_base_module_t *module);

int mca_coll_nccl_reduce(void *sbuf, void *rbuf, int count,
                         struct ompi_datatype_t *dtype,
                         struct ompi_op_t *op,
                         int root,
                         struct ompi_communicator_t *comm,
                         mca_coll_base_module_t *module);
int mca_coll_nccl_bcast(void *buff, int count,
                        struct ompi_datatype_t *datatype,
                        int root,
                        struct ompi_communicator_t *comm,
                        mca_coll_base_module_t *module);

/* Types */
/* Module */

typedef struct mca_coll_nccl_module_t {
    mca_coll_base_module_t super;

    /* Pointers to all the "real" collective functions */
    mca_coll_base_comm_coll_t c_coll;

    /* Pointer to the NCCL communicator */

    ncclComm_t nccl_comm;

    /* Pointer to the internode communicator */
    ompi_communicator_t * intercomm;

    /* My rank in the intranode communicator */
    int rank;

    /* Number of ranks on my local node */
    int node_size;

    /* Do all nodes have the same number of ranks? */
    int balanced;

    /* Node leader rank */
    int leader;

    /* Buffers used in multi node communication */
    char * buffer[2];

    /* Buffers for MPI and NCCL pipelining */
    char * pipeline_buffer[2];

    /* CUDA streams */
    cudaStream_t nccl_stream;
    cudaStream_t op_stream;

    /* Pointers to topology information */
    int * hosts;
    int * nccl_ranks;
    int * intercomm_ranks;
} mca_coll_nccl_module_t;

OBJ_CLASS_DECLARATION(mca_coll_nccl_module_t);

/* Component */

typedef struct mca_coll_nccl_component_t {
    mca_coll_base_component_2_0_0_t super;

    /* Parameters */
    int priority; /* Priority of this component */
    size_t treshold; /* Treshold for using NCCL */
    int verbose; /* Level of verbosity */
    size_t segment_size; /*Segment size for multi node communication */
    size_t pipeline_segment_size; /*Segment size for pipelining multi node communication and NCCL */
} mca_coll_nccl_component_t;

/* Globally exported variables */

OMPI_MODULE_DECLSPEC extern mca_coll_nccl_component_t mca_coll_nccl_component;

END_C_DECLS

#endif /* MCA_COLL_CUDA_EXPORT_H */
