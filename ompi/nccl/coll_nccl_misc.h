/*
 * Copyright (c) 2015      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#ifndef COLL_NCCL_MISC_H
#define COLL_NCCL_MISC_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include "mpi.h"
#include "ompi/communicator/communicator.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/op/op.h"
#include "coll_nccl.h"

ncclResult_t ncclInit(ompi_communicator_t * comm, ncclComm_t * newComm, mca_coll_nccl_module_t * module, int my_rank);

ncclDataType_t convert_type_MPI_to_NCCL(ompi_datatype_t * mpiType);

ncclRedOp_t convert_op_MPI_to_NCCL(ompi_op_t * op);

ncclResult_t check_msg_for_nccl(void *sbuf, void *rbuf, int count, ompi_datatype_t *dtype, ompi_op_t *op, ompi_communicator_t *comm, mca_coll_nccl_module_t *module, ncclDataType_t * nccl_type, ncclRedOp_t * nccl_op);
#endif
