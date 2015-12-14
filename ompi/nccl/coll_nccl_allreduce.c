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

#include "ompi_config.h"
#include "coll_nccl.h"

#include <stdio.h>
#include <math.h>

#include "ompi/op/op.h"
#include "opal/datatype/opal_convertor.h"
#include "opal/datatype/opal_datatype_cuda.h"
#include "ompi/datatype/ompi_datatype.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include "coll_nccl_misc.h"
#include "coll_nccl_debug.h"
#include "coll_nccl_multicoll.h"
#include "coll_nccl_misc.h"


/*
 *	allreduce
 *
 *	Function:	- allreduce using NCCL
 *	Accepts:	- same as MPI_Allreduce()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_nccl_allreduce(void *sbuf, void *rbuf, int count,
                        struct ompi_datatype_t *dtype,
                        struct ompi_op_t *op,
                        struct ompi_communicator_t *comm,
                        mca_coll_base_module_t *module)
{
    int rc = MPI_SUCCESS;
    cudaError_t cuda_error;
    mca_coll_nccl_module_t *s = (mca_coll_nccl_module_t*) module;
    ncclDataType_t nccl_type;
    ncclRedOp_t nccl_op;
    void * sbuf_tmp;
#ifdef NCCL_PROFILE
    struct timespec tnccl1, tnccl2, tallr1, tallr2, tall1, tall2;
    double time_nccl = 0, time_mpi = 0;
#endif

    sbuf_tmp = sbuf == MPI_IN_PLACE? rbuf : sbuf;
    if(check_msg_for_nccl(sbuf_tmp, rbuf, count, dtype, op, comm, s, &nccl_type, &nccl_op) != ncclSuccess)
        goto fallback;
    if(s->intercomm == NULL)
    {
        NCCL_TIMING(tall1);
        ncclAllReduce(sbuf_tmp, rbuf, count, nccl_type, nccl_op, s->nccl_comm, s->nccl_stream);
        cuda_error = cudaStreamSynchronize(s->nccl_stream);
        NCCL_TIMING(tall2);
    }
    else
    {
        size_t msg_size;
        ptrdiff_t datatype_size;
        void * tmp_sbuf[2], * tmp_rbuf[2];
        size_t phase_count[2];
        int current = 0;

        NCCL_TIMING(tall1);
        ompi_datatype_type_extent(dtype, &datatype_size);
        msg_size = datatype_size * count;
        const size_t pss = mca_coll_nccl_component.pipeline_segment_size;
        size_t phase_num = (msg_size + pss - 1) / pss;
        phase_count[current] = (size_t)count < pss/datatype_size?count:pss/datatype_size;
        tmp_sbuf[current] = sbuf_tmp;
        tmp_rbuf[current] = rbuf;
        if(s->balanced && (phase_count[current] % s->node_size) == 0)
        {
            int scatter_size = phase_count[current]/s->node_size;
            size_t scatter_msg_size = scatter_size * datatype_size;
            NCCL_TIMING(tnccl1);
            ncclReduceScatter(tmp_sbuf[current],tmp_rbuf[current] + s->rank * scatter_msg_size, scatter_size, nccl_type, nccl_op, s->nccl_comm, s->nccl_stream);
        }
        else
        {
            NCCL_TIMING(tnccl1);
            ncclReduce(tmp_sbuf[current], tmp_rbuf[current], phase_count[current], nccl_type, nccl_op, 0, s->nccl_comm, s->nccl_stream);
        }
        for(size_t i = 0; i < phase_num; ++i)
        {
            current = current ^ 1;
            cuda_error = cudaStreamSynchronize(s->nccl_stream);
            NCCL_TIMING_ACC(tnccl1, tnccl2, time_nccl);
            if(cuda_error != CUDA_SUCCESS)
            {
                NCCL_VERBOSE(1,"CUDA error: %s, fallback\n", cudaGetErrorString(cuda_error));
                goto fallback;
            }
            /* NCCL */
            if(i != 0)
            {
                if(s->balanced && (phase_count[current] % s->node_size) == 0)
                {
                    NCCL_TIMING(tnccl1);
                    int scatter_size = phase_count[current]/s->node_size;
                    size_t scatter_msg_size = scatter_size * datatype_size;
                    ncclAllGather(tmp_rbuf[current] + s->rank * scatter_msg_size, scatter_size, nccl_type, tmp_rbuf[current], s->nccl_comm, s->nccl_stream);
                }
                else
                {
                    NCCL_TIMING(tnccl1);
                    ncclBcast(tmp_rbuf[current], phase_count[current], nccl_type, 0, s->nccl_comm, s->nccl_stream);
                }
            }
            tmp_sbuf[current] = tmp_sbuf[current ^ 1] + phase_count[current ^ 1] * datatype_size;
            tmp_rbuf[current] = tmp_rbuf[current ^ 1] + phase_count[current ^ 1] * datatype_size;
            int count_left = count - (i+1) * pss / datatype_size;
            phase_count[current] = (size_t)(count_left) < pss/datatype_size?count_left:pss/datatype_size;
            if(i != phase_num - 1)
            {
                if(s->balanced && (phase_count[current] % s->node_size) == 0)
                {
                    int scatter_size = phase_count[current]/s->node_size;
                    size_t scatter_msg_size = scatter_size * datatype_size;
                    NCCL_TIMING(tnccl1);
                    ncclReduceScatter(tmp_sbuf[current],tmp_rbuf[current] + s->rank * scatter_msg_size, scatter_size, nccl_type, nccl_op, s->nccl_comm, s->nccl_stream);
                }
                else
                {
                    NCCL_TIMING(tnccl1);
                    ncclReduce(tmp_sbuf[current], tmp_rbuf[current], phase_count[current], nccl_type, nccl_op, 0, s->nccl_comm, s->nccl_stream);
                }
            }
            /* MPI */
            if(s->balanced && (phase_count[current ^ 1] % s->node_size) == 0)
            {
                int scatter_size = phase_count[current ^ 1]/s->node_size;
                size_t scatter_msg_size = scatter_size * datatype_size;
                NCCL_TIMING(tallr1);
                rc = mca_coll_nccl_multi_allreduce(MPI_IN_PLACE, tmp_rbuf[current ^ 1] + s->rank * scatter_msg_size, scatter_size, dtype, convert_op_MPI_to_NCCL(op), s->intercomm, s);
                NCCL_TIMING_ACC(tallr1, tallr2, time_mpi);
            }
            else
            {
                NCCL_TIMING(tallr1);
                if(s->rank == 0)
                {
                    rc = mca_coll_nccl_multi_allreduce(MPI_IN_PLACE, tmp_rbuf[current ^ 1], phase_count[current ^ 1], dtype, convert_op_MPI_to_NCCL(op), s->intercomm, s);
                }
                NCCL_TIMING_ACC(tallr1, tallr2, time_mpi);
            }
        }
        current = current ^ 1;

        if(s->balanced && (phase_count[current] % s->node_size) == 0)
        {
            NCCL_TIMING(tnccl1);
            int scatter_size = phase_count[current]/s->node_size;
            size_t scatter_msg_size = scatter_size * datatype_size;
            ncclAllGather(tmp_rbuf[current] + s->rank * scatter_msg_size, scatter_size, nccl_type, tmp_rbuf[current], s->nccl_comm, s->nccl_stream);
        }
        else
        {
            NCCL_TIMING(tnccl1);
            ncclBcast(tmp_rbuf[current], phase_count[current], nccl_type, 0, s->nccl_comm, s->nccl_stream);
        }
        cuda_error = cudaStreamSynchronize(s->nccl_stream);
        NCCL_TIMING(tall2);
    }
#ifdef NCCL_PROFILE
    if(s->intercomm == NULL)
    {
        NCCL_VERBOSE(1, "Timing data: %f total\n", get_time(tall1, tall2));
    }
    else
    {
        NCCL_VERBOSE(1, "Timing data: %f total, %f nccl, %f mpi-allred,\n", get_time(tall1, tall2), time_nccl, time_mpi);
    }
#endif
    if(cuda_error == CUDA_SUCCESS)
    {
        return rc;
    }
    NCCL_VERBOSE(1,"CUDA error: %s, fallback\n", cudaGetErrorString(cuda_error));
fallback:
    rc = s->c_coll.coll_allreduce(sbuf, rbuf, count, dtype, op, comm, s->c_coll.coll_allreduce_module);
    return rc;
}

