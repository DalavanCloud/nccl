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
 *	reduce
 *
 *	Function:	- reduce using NCCL
 *	Accepts:	- same as MPI_Reduce()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_nccl_reduce(void *sbuf, void *rbuf, int count,
                     struct ompi_datatype_t *dtype,
                     struct ompi_op_t *op,
                     int root,
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
    struct timespec tnccl1, tnccl2, tmpi1, tmpi2, tall1, tall2;
    double time_nccl = 0, time_mpi = 0;
#endif

    sbuf_tmp = sbuf == MPI_IN_PLACE? rbuf : sbuf;
    if(check_msg_for_nccl(sbuf_tmp, rbuf, count, dtype, op, comm, s, &nccl_type, &nccl_op) != ncclSuccess)
        goto fallback;
    if(s->intercomm == NULL)
    {
        NCCL_TIMING(tall1);
        ncclReduce(sbuf_tmp, rbuf, count, nccl_type, nccl_op, root, s->nccl_comm, s->nccl_stream);
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
        int node_leader = s->balanced?s->nccl_ranks[root]:0;
        int my_rank = ompi_comm_rank(comm);

        NCCL_TIMING(tall1);
        ompi_datatype_type_extent(dtype, &datatype_size);
        msg_size = datatype_size * count;
        const size_t pss = mca_coll_nccl_component.pipeline_segment_size;
        size_t phase_num = (msg_size + pss - 1) / pss;
        phase_count[current] = (size_t)count < pss/datatype_size?count:pss/datatype_size;
        tmp_sbuf[current] = sbuf_tmp;
        tmp_rbuf[current] = my_rank == root?rbuf:s->pipeline_buffer[current];

        NCCL_TIMING(tnccl1);
        ncclReduce(tmp_sbuf[current],tmp_rbuf[current], phase_count[current], nccl_type, nccl_op, node_leader, s->nccl_comm, s->nccl_stream);

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
                    if(!s->balanced && s->hosts[my_rank] == s->hosts[root])
                    {
                        NCCL_TIMING(tnccl1);
                        //This is an inefficient approach - could be achieved by a P2P copy from
                        //the leader to root (which does not exist in NCCL), or removed alltogether
                        //with a different construction of intercomm communicators (which is
                        //impractical - system that would show benefit from that  would need to
                        //have unbalanced number of GPUs and NICs close to all of them.
                        ncclBcast(tmp_rbuf[current], phase_count[current], nccl_type, 0, s->nccl_comm, s->nccl_stream);
                    }
            }
            tmp_sbuf[current] = tmp_sbuf[current ^ 1] + phase_count[current ^ 1] * datatype_size;
            tmp_rbuf[current] = my_rank == root?tmp_rbuf[current ^ 1] + phase_count[current ^ 1] * datatype_size:s->pipeline_buffer[current];
            int count_left = count - (i+1) * pss / datatype_size;
            phase_count[current] = (size_t)(count_left) < pss/datatype_size?count_left:pss/datatype_size;
            if(i != phase_num - 1)
            {
                if(s->balanced || s->hosts[my_rank] != s->hosts[root])
                    NCCL_TIMING(tnccl1);
                ncclReduce(tmp_sbuf[current], tmp_rbuf[current], phase_count[current], nccl_type, nccl_op, node_leader, s->nccl_comm, s->nccl_stream);
            }
            /* MPI */
            NCCL_TIMING(tmpi1);
            if(s->rank == node_leader)
            {
                if(my_rank == root)
                    rc = mca_coll_nccl_multi_reduce(MPI_IN_PLACE, tmp_rbuf[current ^ 1], phase_count[current ^ 1], dtype, convert_op_MPI_to_NCCL(op), s->intercomm_ranks[root], s->intercomm, s);
                else
                    rc = mca_coll_nccl_multi_reduce(tmp_rbuf[current ^ 1], NULL, phase_count[current ^ 1], dtype, convert_op_MPI_to_NCCL(op), s->intercomm_ranks[root], s->intercomm, s);
            }
            NCCL_TIMING_ACC(tmpi1, tmpi2, time_mpi);
        }
        current = current ^ 1;

        if(!s->balanced && s->hosts[my_rank] == s->hosts[root])
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
    rc = s->c_coll.coll_reduce(sbuf, rbuf, count, dtype, op, root, comm, s->c_coll.coll_reduce_module);
    return rc;
}

