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
 *	bcast
 *
 *	Function:	- broadcast using NCCL
 *	Accepts:	- same as MPI_Bcast()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_nccl_bcast(void * buf, int count,
                    struct ompi_datatype_t *dtype,
                    int root,
                    struct ompi_communicator_t *comm,
                    mca_coll_base_module_t *module)
{
    int rc = MPI_SUCCESS;
    cudaError_t cuda_error;
    mca_coll_nccl_module_t *s = (mca_coll_nccl_module_t*) module;
    ncclDataType_t nccl_type;
    ncclRedOp_t nccl_op;
#ifdef NCCL_PROFILE
    struct timespec tnccl1, tnccl2, tmpi1, tmpi2, tall1, tall2;
    double time_nccl = 0, time_mpi = 0;
#endif
    if(check_msg_for_nccl(buf, NULL, count, dtype, NULL, comm, s, &nccl_type, &nccl_op) != ncclSuccess)
        goto fallback;
    if(s->intercomm == NULL)
    {
        NCCL_TIMING(tall1);
        ncclBcast(buf, count, nccl_type, root, s->nccl_comm, s->nccl_stream);
        cuda_error = cudaStreamSynchronize(s->nccl_stream);
        NCCL_TIMING(tall2);
    }
    else
    {
        size_t msg_size;
        ptrdiff_t datatype_size;
        void * tmp_buf[2];
        size_t phase_count[2];
        int current = 0;
        int node_leader = s->balanced?s->nccl_ranks[root]:0;
        int my_rank = ompi_comm_rank(comm);

        NCCL_TIMING(tall1);
        ompi_datatype_type_extent(dtype, &datatype_size);
        msg_size = datatype_size * count;
        const size_t pss = mca_coll_nccl_component.pipeline_segment_size;
        size_t phase_num = (msg_size + pss - 1) / pss;
        phase_count[current] = (size_t)count < pss/datatype_size?count:pss/datatype_size;
        tmp_buf[current] = buf;

        if(s->balanced || s->hosts[my_rank] != s->hosts[root])
        {
            for(size_t i = 0; i <= phase_num; ++i)
            {
                current = current ^ 1;
                /* NCCL */
                if (i != 0)
                {
                    cuda_error = cudaStreamSynchronize(s->nccl_stream);
                    NCCL_TIMING_ACC(tnccl1, tnccl2, time_nccl);
                    if(cuda_error != CUDA_SUCCESS)
                    {
                        NCCL_VERBOSE(1,"CUDA error: %s, fallback\n", cudaGetErrorString(cuda_error));
                        goto fallback;
                    }
                    NCCL_TIMING(tnccl1);
                    ncclBcast(tmp_buf[current], phase_count[current], nccl_type, node_leader, s->nccl_comm, s->nccl_stream);
                }
                /* MPI */
                if(i < phase_num)
                {
                    NCCL_TIMING(tmpi1);
                    if(s->rank == node_leader)
                    {
                        rc = mca_coll_nccl_multi_bcast(tmp_buf[current ^ 1], phase_count[current ^ 1], dtype, s->intercomm_ranks[root], s->intercomm, s);
                    }
                    NCCL_TIMING_ACC(tmpi1, tmpi2, time_mpi);
                }
                tmp_buf[current] = tmp_buf[current ^ 1] + phase_count[current ^ 1] * datatype_size;
                int count_left = count - (i+1) * pss / datatype_size;
                phase_count[current] = (size_t)(count_left) < pss/datatype_size?count_left:pss/datatype_size;
            }
            cuda_error = cudaStreamSynchronize(s->nccl_stream);
        }
        else /* unbalanced, same host as root*/
        {
            for(size_t i = 0; i <= phase_num; ++i)
            {
                current = current ^ 1;
                /* NCCL */
                if (i < phase_num)
                {
                    cuda_error = cudaStreamSynchronize(s->nccl_stream);
                    NCCL_TIMING_ACC(tnccl1, tnccl2, time_nccl);
                    if(cuda_error != CUDA_SUCCESS)
                    {
                        NCCL_VERBOSE(1,"CUDA error: %s, fallback\n", cudaGetErrorString(cuda_error));
                        goto fallback;
                    }
                    NCCL_TIMING(tnccl1);
                    ncclBcast(tmp_buf[current ^ 1], phase_count[current ^ 1], nccl_type, node_leader, s->nccl_comm, s->nccl_stream);
                }
                /* MPI */
                if(i > 0)
                {
                    NCCL_TIMING(tmpi1);
                    if(s->rank == node_leader)
                    {
                            rc = mca_coll_nccl_multi_bcast(tmp_buf[current], phase_count[current], dtype, s->intercomm_ranks[root], s->intercomm, s);
                    }
                    NCCL_TIMING_ACC(tmpi1, tmpi2, time_mpi);
                }
                tmp_buf[current] = tmp_buf[current ^ 1] + phase_count[current ^ 1] * datatype_size;
                int count_left = count - (i+1) * pss / datatype_size;
                phase_count[current] = (size_t)(count_left) < pss/datatype_size?count_left:pss/datatype_size;
            }
            cuda_error = cudaStreamSynchronize(s->nccl_stream);
        }
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
    rc = s->c_coll.coll_bcast(buf, count, dtype, root, comm, s->c_coll.coll_bcast_module);
    return rc;
}

