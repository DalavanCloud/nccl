/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/
#include "../include/macros.h"
#include "ErrorChecker.h"
#include "TEST_ENV.h"
#include "mpi.h"
#include "mpi_fixture.h"
#include "nccl.h"
TYPED_TEST(mpi_test, ncclReduce_basic) {
    for (int iroot = 0; iroot < TEST_ENV::mpi_size; ++iroot)
        for (auto op : this->RedOps) {
            MPI_Barrier(MPI_COMM_WORLD);
            // nccl
            MNCCL_ASSERT(ncclReduce((const void*)this->buf_send_d,
                                    (void*)this->buf_recv_d, this->count1,
                                    this->ncclDataType, op, iroot, this->comm,
                                    this->stream));
            MCUDA_ASSERT(cudaStreamSynchronize(this->stream));
            MCUDA_ASSERT(cudaMemcpy(this->buf_recv_h.data(), this->buf_recv_d,
                                    this->count1 * sizeof(TypeParam),
                                    cudaMemcpyDeviceToHost));
            // mpi
            MPI_Reduce(this->buf_send_h.data(), this->buf_recv_mpi.data(),
                       this->count1, this->mpiDataType, this->MpiOps.at(op),
                       iroot, MPI_COMM_WORLD);
            // only root rank contains the results. so clear the others.
            if (TEST_ENV::mpi_rank != iroot) {
                this->buf_recv_h.assign(this->buf_recv_h.size(), 0);
                this->buf_recv_mpi.assign(this->buf_recv_mpi.size(), 0);
            }
            EXPECT_NO_FATAL_FAILURE(this->Verify(this->count1));
        }
}
