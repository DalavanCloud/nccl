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

#ifdef HAVE_STRING_H
#include <string.h>
#endif
#include <stdio.h>

#include "coll_nccl.h"

#include "mpi.h"

#include "orte/util/show_help.h"
#include "orte/util/proc_info.h"

#include "ompi/constants.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "coll_nccl.h"
#include "coll_nccl_debug.h"
#include "coll_nccl_module_misc.h"

#include <nccl.h>

/*
 * Local functions
 */

static bool single_node_comm(ompi_communicator_t * comm);
static int single_node_color(ompi_communicator_t * comm, int * size);

/*
 * Module constructor
 */

static void mca_coll_nccl_module_construct(mca_coll_nccl_module_t *module)
{
    memset(&(module->c_coll), 0, sizeof(module->c_coll));
    module->nccl_comm = NULL;
}

static void mca_coll_nccl_module_destruct(mca_coll_nccl_module_t *module)
{
    if(module->c_coll.coll_allreduce_module != NULL)
        OBJ_RELEASE(module->c_coll.coll_allreduce_module);
    if(module->c_coll.coll_bcast_module != NULL)
        OBJ_RELEASE(module->c_coll.coll_bcast_module);
    if(module->c_coll.coll_allgather_module != NULL)
        OBJ_RELEASE(module->c_coll.coll_allgather_module);
    if(module->c_coll.coll_reduce_module != NULL)
        OBJ_RELEASE(module->c_coll.coll_reduce_module);

    if (NULL != module->nccl_comm)
    {
        cudaFree(module->buffer[0]);
        cudaFree(module->buffer[1]);
        cudaFree(module->pipeline_buffer[0]);
        cudaFree(module->pipeline_buffer[1]);

        cudaStreamDestroy(module->nccl_stream);
        cudaStreamDestroy(module->op_stream);

        ncclCommDestroy(module->nccl_comm);
    }
}

OBJ_CLASS_INSTANCE(mca_coll_nccl_module_t, mca_coll_base_module_t,
                   mca_coll_nccl_module_construct,
                   mca_coll_nccl_module_destruct);


/*
 * Initial query function that is invoked during MPI_INIT, allowing
 * this component to disqualify itself if it doesn't support the
 * required level of thread support.
 */
int mca_coll_nccl_init_query(bool enable_progress_threads,
                             bool enable_mpi_threads)
{
    (void) enable_progress_threads;
    (void) enable_mpi_threads;
    /* Nothing to do */

    return OMPI_SUCCESS;
}


/*
 * Invoked when there's a new communicator that has been created.
 * Look at the communicator and decide which set of functions and
 * priority we want to return.
 */
mca_coll_base_module_t *
mca_coll_nccl_comm_query(struct ompi_communicator_t *comm,
                         int *priority)
{
    mca_coll_nccl_module_t *nccl_module;

/*
 * Check whether the new communicator is a single node communicator.
 * If it is multi node, then we should not be chosen.
 */
    if(OMPI_COMM_IS_INTER(comm) || ompi_comm_size(comm) == 1)
    {
        NCCL_VERBOSE(5,"Comm: %s, context id: %u, rank %d, Denied\n", comm->c_name, comm->c_contextid, comm->c_my_rank);
        *priority = 0;
        return NULL;
    }

    nccl_module = OBJ_NEW(mca_coll_nccl_module_t);
    if (NULL == nccl_module) {
        return NULL;
    }
    NCCL_VERBOSE(5,"Comm: %s, context id: %u, rank %d, Approved\n", comm->c_name, comm->c_contextid, comm->c_my_rank);


    *priority = mca_coll_nccl_component.priority;
    nccl_module->intercomm = NULL;
    nccl_module->hosts = NULL;
    nccl_module->nccl_ranks = NULL;
    nccl_module->intercomm_ranks = NULL;
    /* Choose whether to use [intra|inter] */
    nccl_module->super.coll_module_enable = mca_coll_nccl_module_enable;
    nccl_module->super.ft_event = NULL;

    nccl_module->super.coll_allgather  = NULL;
    nccl_module->super.coll_allgatherv = NULL;
    nccl_module->super.coll_allreduce  = mca_coll_nccl_allreduce;
    nccl_module->super.coll_alltoall   = NULL;
    nccl_module->super.coll_alltoallv  = NULL;
    nccl_module->super.coll_alltoallw  = NULL;
    nccl_module->super.coll_barrier    = NULL;
    nccl_module->super.coll_bcast      = mca_coll_nccl_bcast;
    nccl_module->super.coll_exscan     = NULL;
    nccl_module->super.coll_gather     = NULL;
    nccl_module->super.coll_gatherv    = NULL;
    nccl_module->super.coll_reduce     = mca_coll_nccl_reduce;
    nccl_module->super.coll_reduce_scatter = NULL;
    nccl_module->super.coll_reduce_scatter_block = NULL;
    nccl_module->super.coll_scan       = NULL;
    nccl_module->super.coll_scatter    = NULL;
    nccl_module->super.coll_scatterv   = NULL;

    return &(nccl_module->super);
}


/*
 * Init module on the communicator
 */
int mca_coll_nccl_module_enable(mca_coll_base_module_t *module,
                                struct ompi_communicator_t *comm)
{
    bool good = true;
    char *msg = NULL;
    mca_coll_nccl_module_t *s = (mca_coll_nccl_module_t*) module;
    int color = -1;
    int count;

    if(mca_coll_nccl_component.priority == 0)
        return OMPI_SUCCESS;

#define CHECK_AND_RETAIN(src, dst, name)               \
    if (NULL == (src)->c_coll.coll_ ## name ## _module) {   \
        good = false; \
        msg = #name; \
    } else if (good) { \
        (dst)->c_coll.coll_ ## name ## _module = (src)->c_coll.coll_ ## name ## _module;\
        (dst)->c_coll.coll_ ## name = (src)->c_coll.coll_ ## name ; \
        OBJ_RETAIN((src)->c_coll.coll_ ## name ## _module); \
    }

    CHECK_AND_RETAIN(comm, s, bcast);
    CHECK_AND_RETAIN(comm, s, allreduce);
    CHECK_AND_RETAIN(comm, s, allgather);
    CHECK_AND_RETAIN(comm, s, reduce);


    s->buffer[0] = NULL;
    s->buffer[1] = NULL;
    s->pipeline_buffer[0] = NULL;
    s->pipeline_buffer[1] = NULL;

    /* Multi node communicator */
    if(!single_node_comm(comm))
    {
        int maxcount, mincount;
        int size = ompi_comm_size(comm);
        int ret;
        color = single_node_color(comm, &count);
        s->node_size = count;
        s->c_coll.coll_allreduce(&count, &maxcount, 1, MPI_INT, MPI_MAX, comm, s->c_coll.coll_allreduce_module);
        s->c_coll.coll_allreduce(&count, &mincount, 1, MPI_INT, MPI_MIN, comm, s->c_coll.coll_allreduce_module);
        if(mincount == 1)
        {
            /* If there is only one rank per node, do not use NCCL */
            NCCL_VERBOSE(1, "Only 1 rank per node in communicator %s, will not use NCCL", comm->c_name);
            return OMPI_ERROR;
        }
        s->hosts = malloc(size * sizeof(int));
        s->nccl_ranks = malloc(size * sizeof(int));
        s->intercomm_ranks = malloc(size * sizeof(int));
        int my_rank = ompi_comm_rank(comm);
        s->c_coll.coll_allgather(&color, 1, MPI_INT, s->hosts, 1, MPI_INT, comm, s->c_coll.coll_allgather_module);
        int intercolor;
        mca_coll_nccl_populate_rankinfo(s->hosts, s->nccl_ranks, s->intercomm_ranks, &(s->leader), &intercolor, color, my_rank, size);
        s->rank = intercolor;
        if(maxcount == mincount)
        {
            s->balanced = 1;
            NCCL_VERBOSE(5, "Multinode communicator is balanced - each node has %d ranks", maxcount);
            ret = ompi_comm_split(comm, intercolor, my_rank, &(s->intercomm), 0);
            if(ret != OMPI_SUCCESS)
            {
                return ret;
            }
            if(OMPI_COMM_CID_IS_LOWER(s->intercomm, comm))
            {
                OMPI_COMM_SET_EXTRA_RETAIN(s->intercomm);
                OBJ_RETAIN(s->intercomm);
            }
        }
        else
        {
            s->balanced = 0;
            NCCL_VERBOSE(5, "Multinode communicator is not balanced - each node has number of ranks between %d and %d", mincount, maxcount);
            int split_color;
            if(intercolor == 0)
                split_color = 0;
            else
                split_color = MPI_UNDEFINED;
            ret = ompi_comm_split(comm, split_color, my_rank, &(s->intercomm), 0);
            if(ret != OMPI_SUCCESS)
            {
                return ret;
            }
            if(s->intercomm != MPI_COMM_NULL && OMPI_COMM_CID_IS_LOWER(s->intercomm, comm))
            {
                OMPI_COMM_SET_EXTRA_RETAIN(s->intercomm);
                OBJ_RETAIN(s->intercomm);
            }
        }
    }
    else
    {
        /* We are on a single node */
        s->rank = ompi_comm_rank(comm);
        s->intercomm = NULL;
        s->balanced = 0;
        s->node_size = ompi_comm_size(comm);
    }

    /* All done */
    if (good) {
        return OMPI_SUCCESS;
    } else {
        orte_show_help("help-mpi-coll-nccl.txt", "missing collective", true,
                       orte_process_info.nodename,
                       mca_coll_nccl_component.priority, msg);
        return OMPI_ERR_NOT_FOUND;
    }
}

static bool single_node_comm(ompi_communicator_t * comm)
{
    int i;
    ompi_proc_t * p;

    for(i = 0; i < ompi_comm_size(comm); ++i)
    {
        p = ompi_group_peer_lookup(comm->c_local_group, i);
        if (!OPAL_PROC_ON_LOCAL_NODE(p->proc_flags)) {
            return false;
        }
    }

    return true;
}

static int single_node_color(ompi_communicator_t * comm, int * size)
{
    int i;
    ompi_proc_t * p;
    int color = -1;
    int count = 0;

    for(i = 0; i < ompi_comm_size(comm); ++i)
    {
        p = ompi_group_peer_lookup(comm->c_local_group, i);
        if(OPAL_PROC_ON_LOCAL_NODE(p->proc_flags))
        {
            count++;
            if(color == -1)
                color = i;
        }
    }
    *size = count;
    return color;
}

