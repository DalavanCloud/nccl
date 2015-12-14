/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009      University of Houston. All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All Rights
 *                         reserved.
 * Copyright (c) 2015      Intel, Inc. All rights reserved.
 * Copyright (c) 2015      NVIDIA Corporation. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "opal/util/bit_ops.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"

#include "coll_nccl.h"
#include "coll_nccl_debug.h"
#include "coll_nccl_multicoll.h"
#include "coll_nccl_kernels.h"

int mca_coll_nccl_multi_sendrecv( void* sendbuf, size_t scount,
                                  ompi_datatype_t* sdatatype,
                                  int dest, int stag,
                                  void* recvbuf, size_t rcount,
                                  ompi_datatype_t* rdatatype,
                                  int source, int rtag,
                                  struct ompi_communicator_t* comm,
                                  ompi_status_public_t* status )

{
    int err, line = 0, nreqs = 0;
    size_t typesize;
    ompi_request_t* reqs[2], **req = reqs;
    ompi_status_public_t statuses[2];

    /* post new irecv */
    ompi_datatype_type_size(rdatatype, &typesize);
    if (0 != rcount && 0 != typesize) {
        err = MCA_PML_CALL(irecv( recvbuf, rcount, rdatatype, source, rtag,
                                  comm, req++));
        ++nreqs;
        if (err != MPI_SUCCESS) { line = __LINE__; goto error_handler; }
    }

    /* send data to children */
    ompi_datatype_type_size(sdatatype, &typesize);
    if (0 != scount && 0 != typesize) {
        err = MCA_PML_CALL(isend( sendbuf, scount, sdatatype, dest, stag,
                                  MCA_PML_BASE_SEND_SYNCHRONOUS, comm, req++));
        ++nreqs;
        if (err != MPI_SUCCESS) { line = __LINE__; goto error_handler; }
    }

    if (0 != nreqs) {
        err = ompi_request_wait_all( nreqs, reqs, statuses );
        if (err != MPI_SUCCESS) { line = __LINE__; goto error_handler; }

        if (MPI_STATUS_IGNORE != status) {
            *status = statuses[0];
        }
    } else {
        if( MPI_STATUS_IGNORE != status )
            *status = ompi_status_empty;
    }

    return (MPI_SUCCESS);

 error_handler:
    /* As we use wait_all we will get MPI_ERR_IN_STATUS which is not an error
     * code that we can propagate up the stack. Instead, look for the real
     * error code from the MPI_ERROR in the status.
     */
    if( MPI_ERR_IN_STATUS == err ) {
        /* At least we know the error was detected during the wait_all */
        int err_index = 1;
        if( MPI_SUCCESS == statuses[0].MPI_ERROR ) {
            err_index = 0;
        }
        if (MPI_STATUS_IGNORE != status) {
            *status = statuses[err_index];
        }
        err = statuses[err_index].MPI_ERROR;
        NCCL_VERBOSE(1,"%s:%d: Error %d occurred in the %s"
                     " stage of mca_coll_nccl_multi_sendrecv\n",
                     __FILE__, line, err, (0 == err_index ? "receive" : "send"));
    } else {
        /* Error discovered during the posting of the irecv or isend,
         * and no status is available.
         */
        NCCL_VERBOSE(1,"%s:%d: Error %d occurred\n",
                     __FILE__, line, err);
        if (MPI_STATUS_IGNORE != status) {
            status->MPI_ERROR = err;
        }
    }
    return (err);
}

/*
 *   mca_coll_nccl_multi_allreduce_ring
 *
 *   Function:       Ring algorithm for allreduce operation
 *   Accepts:        Same as MPI_Allreduce(), NCCL type for operation
 *   Returns:        MPI_SUCCESS or error code
 *
 *   Description:    Ring algorithm for allreduce,
 *                   improved version of the function
 *                   ompi_coll_tuned_allreduce_intra_ring
 *
 */
int
mca_coll_nccl_multi_allreduce_ring(void *sbuf, void *rbuf, int count,
                                         struct ompi_datatype_t *dtype,
                                         ncclRedOp_t op,
                                         struct ompi_communicator_t *comm,
                                         struct mca_coll_nccl_module_t * module)
{
    int ret, line, rank, size, k, recv_from, send_to, block_count, inbi;
    int early_segcount, late_segcount, split_rank, max_segcount;
    size_t typelng;
    char *tmpsend = NULL, *tmprecv = NULL;
    char *inbuf[2] = {module->buffer[0], module->buffer[1]};
    ptrdiff_t true_lb, true_extent, lb, extent;
    ptrdiff_t block_offset, max_real_segsize;
    ompi_request_t *reqs[2] = {NULL, NULL};
#ifdef NCCL_PROFILE
    struct timespec tredsc1, tredsc2, top1, top2, tallg1, tallg2;
    double time_op = 0;
#endif
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    /* Allocate and initialize temporary buffers */
    ret = ompi_datatype_get_extent(dtype, &lb, &extent);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    ret = ompi_datatype_get_true_extent(dtype, &true_lb, &true_extent);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    ret = ompi_datatype_type_size( dtype, &typelng);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    /* Determine the number of elements per block and corresponding
       block sizes.
       The blocks are divided into "early" and "late" ones:
       blocks 0 .. (split_rank - 1) are "early" and
       blocks (split_rank) .. (size - 1) are "late".
       Early blocks are at most 1 element larger than the late ones.
    */
    COLL_NCCL_MULTI_COMPUTE_BLOCKCOUNT( count, size, split_rank,
                                   early_segcount, late_segcount )
        max_segcount = early_segcount;
    max_real_segsize = true_extent + (max_segcount - 1) * extent;

    /* Handle MPI_IN_PLACE */
    if (MPI_IN_PLACE != sbuf) {
        NCCL_VERBOSE(1, "Not implemented out-of-place version!\n");
        return MPI_ERR_UNSUPPORTED_OPERATION;
    }

    /* Computation loop */

    /*
       For each of the remote nodes:
       - post irecv for block (r-1)
       - send block (r)
       - in loop for every step k = 2 .. n
       - post irecv for block (r + n - k) % n
       - wait on block (r + n - k + 1) % n to arrive
       - compute on block (r + n - k + 1) % n
       - send block (r + n - k + 1) % n
       - wait on block (r + 1)
       - compute on block (r + 1)
       - send block (r + 1) to rank (r + 1)
       Note that we must be careful when computing the begining of buffers and
       for send operations and computation we must compute the exact block size.
    */
    send_to = (rank + 1) % size;
    recv_from = (rank + size - 1) % size;

    inbi = 0;
    NCCL_TIMING(tredsc1);
    /* Initialize first receive from the neighbor on the left */
    ret = MCA_PML_CALL(irecv(inbuf[inbi], max_segcount, dtype, recv_from,
                             MCA_COLL_BASE_TAG_ALLREDUCE, comm, &reqs[inbi]));
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    /* Send first block (my block) to the neighbor on the right */
    block_offset = ((rank < split_rank)?
                    ((ptrdiff_t)rank * (ptrdiff_t)early_segcount) :
                    ((ptrdiff_t)rank * (ptrdiff_t)late_segcount + split_rank));
    block_count = ((rank < split_rank)? early_segcount : late_segcount);
    tmpsend = ((char*)rbuf) + block_offset * extent;
    ret = MCA_PML_CALL(send(tmpsend, block_count, dtype, send_to,
                            MCA_COLL_BASE_TAG_ALLREDUCE,
                            MCA_PML_BASE_SEND_SYNCHRONOUS, comm));
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    for (k = 2; k < size; k++) {
        const int prevblock = (rank + size - k + 1) % size;

        inbi = inbi ^ 0x1;

        /* Post irecv for the current block */
        ret = MCA_PML_CALL(irecv(inbuf[inbi], max_segcount, dtype, recv_from,
                                 MCA_COLL_BASE_TAG_ALLREDUCE, comm, &reqs[inbi]));
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

        /* Wait on previous block to arrive */
        ret = ompi_request_wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

        /* Apply operation on previous block: result goes to rbuf
           rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
        */
        block_offset = ((prevblock < split_rank)?
                        ((ptrdiff_t)prevblock * early_segcount) :
                        ((ptrdiff_t)prevblock * late_segcount + split_rank));
        block_count = ((prevblock < split_rank)? early_segcount : late_segcount);
        tmprecv = ((char*)rbuf) + (ptrdiff_t)block_offset * extent;
        NCCL_TIMING(top1);
        mca_coll_nccl_cuda_op_reduce(tmprecv, inbuf[inbi ^ 0x1], block_count, dtype, op, module->op_stream);
        NCCL_TIMING_ACC(top1, top2, time_op);

        /* send previous block to send_to */
        ret = MCA_PML_CALL(send(tmprecv, block_count, dtype, send_to,
                                MCA_COLL_BASE_TAG_ALLREDUCE,
                                MCA_PML_BASE_SEND_SYNCHRONOUS, comm));
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    }

    /* Wait on the last block to arrive */
    ret = ompi_request_wait(&reqs[inbi], MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    /* Apply operation on the last block (from neighbor (rank + 1)
       rbuf[rank+1] = inbuf[inbi] (op) rbuf[rank + 1] */
    recv_from = (rank + 1) % size;
    block_offset = ((recv_from < split_rank)?
                    ((ptrdiff_t)recv_from * early_segcount) :
                    ((ptrdiff_t)recv_from * late_segcount + split_rank));
    block_count = ((recv_from < split_rank)? early_segcount : late_segcount);
    tmprecv = ((char*)rbuf) + (ptrdiff_t)block_offset * extent;
    NCCL_TIMING(top1);
    mca_coll_nccl_cuda_op_reduce(tmprecv, inbuf[inbi], block_count, dtype, op, module->op_stream);
    NCCL_TIMING_ACC(top1, top2, time_op);
    NCCL_TIMING(tredsc2);
    NCCL_TIMING(tallg1);
    /* Distribution loop - variation of ring allgather */
    send_to = (rank + 1) % size;
    recv_from = (rank + size - 1) % size;
    for (k = 0; k < size - 1; k++) {
        const int recv_data_from = (rank + size - k) % size;
        const int send_data_from = (rank + 1 + size - k) % size;
        const int send_block_offset =
            ((send_data_from < split_rank)?
             ((ptrdiff_t)send_data_from * early_segcount) :
             ((ptrdiff_t)send_data_from * late_segcount + split_rank));
        const int recv_block_offset =
            ((recv_data_from < split_rank)?
             ((ptrdiff_t)recv_data_from * early_segcount) :
             ((ptrdiff_t)recv_data_from * late_segcount + split_rank));
        block_count = ((send_data_from < split_rank)?
                       early_segcount : late_segcount);

        tmprecv = (char*)rbuf + (ptrdiff_t)recv_block_offset * extent;
        tmpsend = (char*)rbuf + (ptrdiff_t)send_block_offset * extent;

        ret = mca_coll_nccl_multi_sendrecv(tmpsend, block_count, dtype, send_to,
                                           MCA_COLL_BASE_TAG_ALLREDUCE,
                                           tmprecv, max_segcount, dtype, recv_from,
                                           MCA_COLL_BASE_TAG_ALLREDUCE,
                                           comm, MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl;}

    }
    NCCL_TIMING(tallg2);
#ifdef NCCL_PROFILE
    NCCL_VERBOSE(1, "Timing data (inter node): %f reduce-scatter, including %f operation, %f allgather\n", get_time(tredsc1, tredsc2), time_op, get_time(tallg1, tallg2));
#endif

    return MPI_SUCCESS;

 error_hndl:
    NCCL_VERBOSE(1,"%s:%4d\tRank %d Error occurred %d\n",
                        __FILE__, line, rank, ret);
    return ret;
}

/*
 *   mca_coll_nccl_multi_allreduce_ring_segmented
 *
 *   Function:       Pipelined ring algorithm for allreduce operation
 *   Accepts:        Same as MPI_Allreduce(), NCCL type for operation, segment size
 *   Returns:        MPI_SUCCESS or error code
 *
 *   Description:    Ring algorithm for allreduce,
 *                   improved version of the function
 *                   ompi_coll_tuned_allreduce_intra_ring_segmented
 *
 */
int
mca_coll_nccl_multi_allreduce_ring_segmented(void *sbuf, void *rbuf, int count,
                                             struct ompi_datatype_t *dtype,
                                             ncclRedOp_t op,
                                             struct ompi_communicator_t *comm,
                                             struct mca_coll_nccl_module_t *module,
                                             size_t segsize)
{
    int ret, line, rank, size, k, recv_from, send_to;
    int early_blockcount, late_blockcount, split_rank;
    int segcount, max_segcount, num_phases, phase, block_count, inbi;
    size_t typelng;
    char *tmpsend = NULL, *tmprecv = NULL;
    char *inbuf[2] = {module->buffer[0], module->buffer[1]};
    ptrdiff_t lb, extent;
    ptrdiff_t block_offset;
    ompi_request_t *reqs[2] = {NULL, NULL};
#ifdef NCCL_PROFILE
    struct timespec tredsc1, tredsc2, top1, top2, tallg1, tallg2;
    double time_op = 0;
#endif

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    /* Determine segment count based on the suggested segment size */
    ret = ompi_datatype_get_extent(dtype, &lb, &extent);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    ret = ompi_datatype_type_size( dtype, &typelng);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    segcount = count;
    COLL_NCCL_MULTI_COMPUTED_SEGCOUNT(segsize, typelng, segcount)

    /* Determine the number of phases of the algorithm */
    num_phases = count / (size * segcount);
    if (count % (size * segcount) >= size) {
        num_phases++;
    }

    /* Determine the number of elements per block and corresponding
       block sizes.
       The blocks are divided into "early" and "late" ones:
       blocks 0 .. (split_rank - 1) are "early" and
       blocks (split_rank) .. (size - 1) are "late".
       Early blocks are at most 1 element larger than the late ones.
       Note, these blocks will be split into num_phases segments,
       out of the largest one will have max_segcount elements.
    */
    COLL_NCCL_MULTI_COMPUTE_BLOCKCOUNT( count, size, split_rank,
                                        early_blockcount, late_blockcount );
    COLL_NCCL_MULTI_COMPUTE_BLOCKCOUNT( early_blockcount, num_phases, inbi,
                                        max_segcount, k);

    /* Handle MPI_IN_PLACE */
    if (MPI_IN_PLACE != sbuf) {
        NCCL_VERBOSE(1, "Out-of-place version not implemented!\n");
        return MPI_ERR_UNSUPPORTED_OPERATION;
    }
    NCCL_TIMING(tredsc1);
    /* Computation loop: for each phase, repeat ring allreduce computation loop */
    for (phase = 0; phase < num_phases; phase ++) {
        ptrdiff_t phase_offset;
        int early_phase_segcount, late_phase_segcount, split_phase, phase_count;

        /*
           For each of the remote nodes:
           - post irecv for block (r-1)
           - send block (r)
           To do this, first compute block offset and count, and use block offset
           to compute phase offset.
           - in loop for every step k = 2 .. n
           - post irecv for block (r + n - k) % n
           - wait on block (r + n - k + 1) % n to arrive
           - compute on block (r + n - k + 1) % n
           - send block (r + n - k + 1) % n
           - wait on block (r + 1)
           - compute on block (r + 1)
           - send block (r + 1) to rank (r + 1)
           Note that we must be careful when computing the begining of buffers and
           for send operations and computation we must compute the exact block size.
        */
        send_to = (rank + 1) % size;
        recv_from = (rank + size - 1) % size;

        inbi = 0;
        /* Initialize first receive from the neighbor on the left */
        ret = MCA_PML_CALL(irecv(inbuf[inbi], max_segcount, dtype, recv_from,
                                 MCA_COLL_BASE_TAG_ALLREDUCE, comm, &reqs[inbi]));
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
        /* Send first block (my block) to the neighbor on the right:
           - compute my block and phase offset
           - send data */
        block_offset = ((rank < split_rank)?
                        ((ptrdiff_t)rank * (ptrdiff_t)early_blockcount) :
                        ((ptrdiff_t)rank * (ptrdiff_t)late_blockcount + split_rank));
        block_count = ((rank < split_rank)? early_blockcount : late_blockcount);
        COLL_NCCL_MULTI_COMPUTE_BLOCKCOUNT(block_count, num_phases, split_phase,
                                      early_phase_segcount, late_phase_segcount)
            phase_count = ((phase < split_phase)?
                           (early_phase_segcount) : (late_phase_segcount));
        phase_offset = ((phase < split_phase)?
                        ((ptrdiff_t)phase * (ptrdiff_t)early_phase_segcount) :
                        ((ptrdiff_t)phase * (ptrdiff_t)late_phase_segcount + split_phase));
        tmpsend = ((char*)rbuf) + (ptrdiff_t)(block_offset + phase_offset) * extent;
        ret = MCA_PML_CALL(send(tmpsend, phase_count, dtype, send_to,
                                MCA_COLL_BASE_TAG_ALLREDUCE,
                                MCA_PML_BASE_SEND_SYNCHRONOUS, comm));
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

        for (k = 2; k < size; k++) {
            const int prevblock = (rank + size - k + 1) % size;

            inbi = inbi ^ 0x1;

            /* Post irecv for the current block */
            ret = MCA_PML_CALL(irecv(inbuf[inbi], max_segcount, dtype, recv_from,
                                     MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                     &reqs[inbi]));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

            /* Wait on previous block to arrive */
            ret = ompi_request_wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

            /* Apply operation on previous block: result goes to rbuf
               rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
            */
            block_offset = ((prevblock < split_rank)?
                            ((ptrdiff_t)prevblock * (ptrdiff_t)early_blockcount) :
                            ((ptrdiff_t)prevblock * (ptrdiff_t)late_blockcount + split_rank));
            block_count = ((prevblock < split_rank)?
                           early_blockcount : late_blockcount);
            COLL_NCCL_MULTI_COMPUTE_BLOCKCOUNT(block_count, num_phases, split_phase,
                                          early_phase_segcount, late_phase_segcount)
                phase_count = ((phase < split_phase)?
                               (early_phase_segcount) : (late_phase_segcount));
            phase_offset = ((phase < split_phase)?
                            ((ptrdiff_t)phase * (ptrdiff_t)early_phase_segcount) :
                            ((ptrdiff_t)phase * (ptrdiff_t)late_phase_segcount + split_phase));
            tmprecv = ((char*)rbuf) + (ptrdiff_t)(block_offset + phase_offset) * extent;
            NCCL_TIMING(top1);
            mca_coll_nccl_cuda_op_reduce(tmprecv, inbuf[inbi ^ 0x1], phase_count, dtype, op, module->op_stream);
            NCCL_TIMING_ACC(top1, top2, time_op);

            /* send previous block to send_to */
            ret = MCA_PML_CALL(send(tmprecv, phase_count, dtype, send_to,
                                    MCA_COLL_BASE_TAG_ALLREDUCE,
                                    MCA_PML_BASE_SEND_SYNCHRONOUS, comm));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
        }

        /* Wait on the last block to arrive */
        ret = ompi_request_wait(&reqs[inbi], MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

        /* Apply operation on the last block (from neighbor (rank + 1)
           rbuf[rank+1] = inbuf[inbi] (op) rbuf[rank + 1] */
        recv_from = (rank + 1) % size;
        block_offset = ((recv_from < split_rank)?
                        ((ptrdiff_t)recv_from * (ptrdiff_t)early_blockcount) :
                        ((ptrdiff_t)recv_from * (ptrdiff_t)late_blockcount + split_rank));
        block_count = ((recv_from < split_rank)?
                       early_blockcount : late_blockcount);
        COLL_NCCL_MULTI_COMPUTE_BLOCKCOUNT(block_count, num_phases, split_phase,
                                      early_phase_segcount, late_phase_segcount)
            phase_count = ((phase < split_phase)?
                           (early_phase_segcount) : (late_phase_segcount));
        phase_offset = ((phase < split_phase)?
                        ((ptrdiff_t)phase * (ptrdiff_t)early_phase_segcount) :
                        ((ptrdiff_t)phase * (ptrdiff_t)late_phase_segcount + split_phase));
        tmprecv = ((char*)rbuf) + (ptrdiff_t)(block_offset + phase_offset) * extent;
        NCCL_TIMING(top1);
        mca_coll_nccl_cuda_op_reduce(tmprecv, inbuf[inbi], phase_count, dtype, op, module->op_stream);
        NCCL_TIMING_ACC(top1, top2, time_op);
    }
    NCCL_TIMING(tredsc2);

    NCCL_TIMING(tallg1);
    /* Distribution loop - variation of ring allgather */
    send_to = (rank + 1) % size;
    recv_from = (rank + size - 1) % size;
    for (k = 0; k < size - 1; k++) {
        const int recv_data_from = (rank + size - k) % size;
        const int send_data_from = (rank + 1 + size - k) % size;
        const int send_block_offset =
            ((send_data_from < split_rank)?
             ((ptrdiff_t)send_data_from * (ptrdiff_t)early_blockcount) :
             ((ptrdiff_t)send_data_from * (ptrdiff_t)late_blockcount + split_rank));
        const int recv_block_offset =
            ((recv_data_from < split_rank)?
             ((ptrdiff_t)recv_data_from * (ptrdiff_t)early_blockcount) :
             ((ptrdiff_t)recv_data_from * (ptrdiff_t)late_blockcount + split_rank));
        block_count = ((send_data_from < split_rank)?
                       early_blockcount : late_blockcount);

        tmprecv = (char*)rbuf + (ptrdiff_t)recv_block_offset * extent;
        tmpsend = (char*)rbuf + (ptrdiff_t)send_block_offset * extent;

        ret = mca_coll_nccl_multi_sendrecv(tmpsend, block_count, dtype, send_to,
                                           MCA_COLL_BASE_TAG_ALLREDUCE,
                                           tmprecv, early_blockcount, dtype, recv_from,
                                           MCA_COLL_BASE_TAG_ALLREDUCE,
                                           comm, MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl;}

    }
    NCCL_TIMING(tallg2);
#ifdef NCCL_PROFILE
    NCCL_VERBOSE(1, "Timing data (inter node): %f reduce-scatter, including %f operation, %f allgather\n", get_time(tredsc1, tredsc2), time_op, get_time(tallg1, tallg2));
#endif

    return MPI_SUCCESS;

 error_hndl:
    NCCL_VERBOSE(1,"%s:%4d\tRank %d Error occurred %d\n",
                        __FILE__, line, rank, ret);
    return ret;
}

int
mca_coll_nccl_multi_allreduce(void *sbuf, void *rbuf, int count,
                              struct ompi_datatype_t *dtype,
                              ncclRedOp_t op,
                              struct ompi_communicator_t *comm,
                              struct mca_coll_nccl_module_t *module)
{
    int comm_size = ompi_comm_size(comm);
    ptrdiff_t dt_size;
    ptrdiff_t lb;
    ompi_datatype_get_extent(dtype, &lb, &dt_size);

    if(mca_coll_nccl_component.segment_size * comm_size > (unsigned long)count * dt_size)
    {
        return mca_coll_nccl_multi_allreduce_ring(sbuf, rbuf, count, dtype, op,
                                                  comm, module);
    }
    else
    {
        return mca_coll_nccl_multi_allreduce_ring_segmented(sbuf, rbuf, count,
                                                            dtype, op, comm,
                                                            module,
                                                            mca_coll_nccl_component.segment_size);
    }
}

/*
 *
 *  mca_coll_nccl_multi_reduce
 *
 *  Function: Ring algorithm for reduce operation
 *  Accepts: Same as MPI_Reduce, NCCL type of operation
 *  Returns: MPI_SUCCESS or error code
 *
 *  Description: Ring algorithm for reduce.
 *
 */

int
mca_coll_nccl_multi_reduce(void *sbuf, void *rbuf, int count,
                           struct ompi_datatype_t *dtype,
                           ncclRedOp_t op,
                           int root,
                           struct ompi_communicator_t *comm,
                           struct mca_coll_nccl_module_t *module)
{
    int my_rank = ompi_comm_rank(comm);
    int size = ompi_comm_size(comm);
    char * inbuf[2] = {module->buffer[0], module->buffer[1]};
    int inbi = 0;
    size_t segsize = mca_coll_nccl_component.segment_size;
    int ret, line;
    ptrdiff_t lb, extent;
    int segcount, num_phases;
    int phase;
    void * tmp_sbuf;
    int first_rank = (root + 1) % size;
    int next_rank = (my_rank + 1) % size;
    int prev_rank = (my_rank + size - 1) % size;
    int phase_count[2];
    ompi_request_t *reqs[2] = {NULL, NULL};


    if(my_rank == root && sbuf != MPI_IN_PLACE)
    {
        NCCL_VERBOSE(1, "Out-of-place version not implemented!\n");
        return MPI_ERR_UNSUPPORTED_OPERATION;
    }

    /* Determine segment count based on the suggested segment size */
    ret = ompi_datatype_get_extent(dtype, &lb, &extent);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    segcount = segsize / extent;

    /* Determine the number of phases of the algorithm */
    num_phases = (count + segcount - 1) / segcount;
    tmp_sbuf = my_rank == root? rbuf : sbuf;
    phase_count[inbi] = count > segcount? segcount : count;

    if(my_rank != first_rank)
    {
        ret = MCA_PML_CALL(irecv(inbuf[inbi], phase_count[inbi], dtype, prev_rank,
                                 MCA_COLL_BASE_TAG_REDUCE, comm, &reqs[inbi]));
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    }

    for(phase = 0; phase < num_phases; ++phase)
    {
        int count_left = count - (phase + 1) * segcount;
        inbi = inbi ^ 1;
        phase_count[inbi] = count_left > segcount ? segcount : count_left;


        if(my_rank != first_rank)
        {
            ret = ompi_request_wait(&reqs[inbi ^ 1], MPI_STATUS_IGNORE);
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

            if(phase < num_phases - 1)
            {
                ret = MCA_PML_CALL(irecv(inbuf[inbi], phase_count[inbi], dtype, prev_rank,
                            MCA_COLL_BASE_TAG_REDUCE, comm, &reqs[inbi]));
                if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
            }

            if(my_rank != root)
            {
                mca_coll_nccl_cuda_op_reduce(inbuf[inbi ^ 1], tmp_sbuf, phase_count[inbi ^ 1], dtype, op, module->op_stream);
            }
            else
            {
                mca_coll_nccl_cuda_op_reduce(tmp_sbuf, inbuf[inbi ^ 1], phase_count[inbi ^ 1], dtype, op, module->op_stream);
            }
            cudaStreamSynchronize(module->op_stream);
        }

        if(my_rank != root && my_rank != first_rank)
        {
            ret = MCA_PML_CALL(send(inbuf[inbi ^ 1], phase_count[inbi ^ 1], dtype, next_rank,
                                    MCA_COLL_BASE_TAG_REDUCE,
                                    MCA_PML_BASE_SEND_SYNCHRONOUS, comm));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
        }
        else if (my_rank == first_rank)
        {
            ret = MCA_PML_CALL(send(tmp_sbuf, phase_count[inbi ^ 1], dtype, next_rank,
                                    MCA_COLL_BASE_TAG_REDUCE,
                                    MCA_PML_BASE_SEND_SYNCHRONOUS, comm));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
        }
        tmp_sbuf += segsize;
    }

    return MPI_SUCCESS;

error_hndl:
    NCCL_VERBOSE(1,"%s:%4d\tRank %d Error occurred %d\n",
                        __FILE__, line, my_rank, ret);
    return ret;
}

/*
 *
 *  mca_coll_nccl_multi_bcast
 *
 *  Function: Ring algorithm for broadcast operation
 *  Accepts: Same as MPI_Bcast
 *  Returns: MPI_SUCCESS or error code
 *
 *  Description: Ring algorithm for broadcast.
 *
 */

int
mca_coll_nccl_multi_bcast(void * buf, int count,
                          struct ompi_datatype_t *dtype,
                          int root,
                          struct ompi_communicator_t *comm,
                          struct mca_coll_nccl_module_t *module)
{
    int my_rank = ompi_comm_rank(comm);
    int size = ompi_comm_size(comm);
    char * inbuf[2] = {NULL, NULL};
    int inbi = 0;
    size_t segsize = mca_coll_nccl_component.segment_size;
    int ret, line;
    ptrdiff_t lb, extent;
    int segcount, num_phases;
    int phase;
    int last_rank = (root + size - 1) % size;
    int next_rank = (my_rank + 1) % size;
    int prev_rank = (my_rank + size - 1) % size;
    int phase_count[2];
    ompi_request_t *reqs[2] = {NULL, NULL};
    (void) module;

    /* Determine segment count based on the suggested segment size */
    ret = ompi_datatype_get_extent(dtype, &lb, &extent);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    segcount = segsize / extent;

    /* Determine the number of phases of the algorithm */
    num_phases = (count + segcount - 1) / segcount;
    inbuf[inbi] = buf;
    phase_count[inbi] = count > segcount? segcount : count;

    if(my_rank != root)
    {
        ret = MCA_PML_CALL(irecv(inbuf[inbi], phase_count[inbi], dtype, prev_rank,
                                 MCA_COLL_BASE_TAG_REDUCE, comm, &reqs[inbi]));
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    }

    for(phase = 0; phase < num_phases; ++phase)
    {
        int count_left = count - (phase + 1) * segcount;
        inbi = inbi ^ 1;
        phase_count[inbi] = count_left > segcount ? segcount : count_left;
        inbuf[inbi] = inbuf[inbi ^ 1] + phase_count[inbi ^ 1] * extent;

        if(my_rank != root)
        {
            ret = ompi_request_wait(&reqs[inbi ^ 1], MPI_STATUS_IGNORE);
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

            if(phase < num_phases - 1)
            {
                ret = MCA_PML_CALL(irecv(inbuf[inbi], phase_count[inbi], dtype, prev_rank,
                            MCA_COLL_BASE_TAG_REDUCE, comm, &reqs[inbi]));
                if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
            }
        }

        if(my_rank != last_rank)
        {
            ret = MCA_PML_CALL(send(inbuf[inbi ^ 1], phase_count[inbi ^ 1], dtype, next_rank,
                                    MCA_COLL_BASE_TAG_REDUCE,
                                    MCA_PML_BASE_SEND_SYNCHRONOUS, comm));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
        }
    }

    return MPI_SUCCESS;

error_hndl:
    NCCL_VERBOSE(1,"%s:%4d\tRank %d Error occurred %d\n",
                        __FILE__, line, my_rank, ret);
    return ret;
}
