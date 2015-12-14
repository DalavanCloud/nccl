/*
 * Copyright (c) 2015      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <unistd.h>

/* From Linux kernel source */
#define PID_MAX_LIMIT (4*1024*1024)

#include "coll_nccl_misc.h"
#include "coll_nccl.h"

#include "ompi/constants.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/datatype/ompi_datatype_internal.h"
#include "coll_nccl.h"

#include "coll_nccl_debug.h"


ncclResult_t ncclInit(ompi_communicator_t * comm, ncclComm_t * newComm, mca_coll_nccl_module_t * module, int my_rank)
{
    ncclUniqueId job_id;

    if(my_rank == 0)
    {
        ncclGetUniqueId(&job_id);
    }
    if(module->intercomm == NULL)
    {
        module->c_coll.coll_bcast((char*)&job_id, sizeof(ncclUniqueId), MPI_CHAR, 0, comm, module->c_coll.coll_bcast_module);
    }
    else
    {
        ncclUniqueId * job_array = (ncclUniqueId *) malloc(ompi_comm_size(comm) * sizeof(ncclUniqueId));
        module->c_coll.coll_allgather(&job_id, sizeof(ncclUniqueId), MPI_CHAR, job_array, sizeof(ncclUniqueId), MPI_CHAR, comm, module->c_coll.coll_allgather_module);
        job_id = job_array[module->leader];
        free(job_array);
    }
    return ncclCommInitRank(newComm, module->node_size, job_id, my_rank);
}

/* We add 1 to all known values, so that if we get 0
 * from the lookup table, than we know that we did not hit
 * anything.
 */
ncclDataType_t type_lookup[OMPI_DATATYPE_MAX_PREDEFINED] = {
    [OMPI_DATATYPE_MPI_CHAR] = ncclChar+1,
    [OMPI_DATATYPE_MPI_INT]  = ncclInt+1,
    /*no predefined MPI type for fp16*/
    [OMPI_DATATYPE_MPI_FLOAT] = ncclFloat+1,
    [OMPI_DATATYPE_MPI_DOUBLE] = ncclDouble+1
};

ncclDataType_t convert_type_MPI_to_NCCL(ompi_datatype_t * mpiType)
{
    if(mpiType->id < OMPI_DATATYPE_MAX_PREDEFINED)
    {
        if(type_lookup[mpiType->id] == 0)
            return nccl_NUM_TYPES;
        return type_lookup[mpiType->id] - 1;
    }
    return nccl_NUM_TYPES;
}

ncclRedOp_t op_lookup[OMPI_OP_NUM_OF_TYPES] = {
    [OMPI_OP_SUM] = ncclSum + 1,
    [OMPI_OP_PROD] = ncclProd + 1,
    [OMPI_OP_MAX] = ncclMax + 1,
    [OMPI_OP_MIN] = ncclMin + 1
};

ncclRedOp_t convert_op_MPI_to_NCCL(ompi_op_t * op)
{
    if(op == NULL)
        return ncclSum;
    if(op->op_type < OMPI_OP_NUM_OF_TYPES)
    {
        if(op_lookup[op->op_type] == 0)
            return nccl_NUM_OPS;
        return op_lookup[op->op_type] - 1;
    }
    return nccl_NUM_OPS;
}


ncclResult_t check_msg_for_nccl(void *sbuf, void *rbuf, int count, ompi_datatype_t *dtype, ompi_op_t *op, ompi_communicator_t *comm, mca_coll_nccl_module_t *module, ncclDataType_t * nccl_type, ncclRedOp_t * nccl_op)
{
    ptrdiff_t extent;
    ompi_datatype_type_extent(dtype, &extent);
    NCCL_VERBOSE(5,"NCCL: Comm %s, rank %d, extent %ld. messageSize %ld\n", comm->c_name, comm->c_my_rank, extent, extent * count);

    if(extent * count < mca_coll_nccl_component.treshold)
    {
        NCCL_VERBOSE(1,"Message too small, fallback\n");
        return ncclSystemError;
    }
    if(mca_coll_nccl_component.pipeline_segment_size % extent != 0)
    {
        NCCL_VERBOSE(1, "Pipeline segment size is not multiple of the used datatype size, fallback\n");
        return ncclInvalidArgument;
    }

    if(module->nccl_comm == NULL)
    {
        cudaMalloc((void**)&(module->buffer[0]), mca_coll_nccl_component.segment_size);
        cudaMalloc((void**)&(module->buffer[1]), mca_coll_nccl_component.segment_size);
        cudaMalloc((void**)&(module->pipeline_buffer[0]), mca_coll_nccl_component.pipeline_segment_size);
        cudaMalloc((void**)&(module->pipeline_buffer[1]), mca_coll_nccl_component.pipeline_segment_size);
        cudaStreamCreate(&(module->nccl_stream));
        cudaStreamCreate(&(module->op_stream));
        if(ncclSuccess != ncclInit(comm, &(module->nccl_comm), module, module->rank))
        {
            NCCL_VERBOSE(1,"Failed to initialize NCCL communicator, fallback\n");
            return ncclSystemError;
        }
    }
    struct cudaPointerAttributes sptr_attrib;
    struct cudaPointerAttributes dptr_attrib;
    cudaError_t cuda_error = cudaSuccess;
    if(sbuf != NULL)
        cuda_error = cudaPointerGetAttributes(&sptr_attrib, sbuf);
    else
        sptr_attrib.memoryType = cudaMemoryTypeDevice;
    if(rbuf != NULL)
        cuda_error |= cudaPointerGetAttributes(&dptr_attrib, rbuf);
    else
        dptr_attrib.memoryType = cudaMemoryTypeDevice;
    cudaGetLastError(); //Reset the possible error
    int bad;
    if(cuda_error == cudaSuccess && sptr_attrib.memoryType == cudaMemoryTypeDevice && dptr_attrib.memoryType == cudaMemoryTypeDevice)
        bad = 0;
    else
        bad = 1;
    int result;
    module->c_coll.coll_allreduce(&bad, &result, 1, MPI_INT, MPI_SUM, comm, module->c_coll.coll_allreduce_module);

    if(result > 0)
    {
        NCCL_VERBOSE(1,"Not all buffers in the GPU memory, fallback\n");
        return ncclInvalidDevicePointer;
    }
    *nccl_type = convert_type_MPI_to_NCCL(dtype);
    if (*nccl_type == nccl_NUM_TYPES)
    {
        NCCL_VERBOSE(1,"Type of messages is not supported by NCCL, fallback\n");
        return ncclSystemError;
    }
    *nccl_op = convert_op_MPI_to_NCCL(op);
    if(*nccl_op == nccl_NUM_OPS)
    {
        NCCL_VERBOSE(1,"Operation not supported by NCCL, fallback\n");
        return ncclSystemError;
    }

    return ncclSuccess;
}
