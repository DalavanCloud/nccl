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
/// TODO: only assert in the root process .
TYPED_TEST(mpi_test, ncclBcast_basic) {
    for (int iroot = 0; iroot < TEST_ENV::mpi_size; ++iroot) {
        MPI_Barrier(MPI_COMM_WORLD);
        // init data
        for (int i = 0; i < this->count1; i++) {
            this->buf_send_h[i] = (TEST_ENV::mpi_rank == iroot)
                                      ? (TEST_ENV::mpi_rank * this->count1 + i)
                                      : 0;
        }
        MCUDA_ASSERT(cudaMemcpy(this->buf_send_d, this->buf_send_h.data(),
                                this->count1 * sizeof(TypeParam),
                                cudaMemcpyHostToDevice));
        MPI_Barrier(MPI_COMM_WORLD);
        // nccl
        MNCCL_ASSERT(ncclBcast((void*)this->buf_send_d, this->count1,
                               this->ncclDataType, iroot, this->comm,
                               this->stream));
        MCUDA_ASSERT(cudaStreamSynchronize(this->stream));
        MCUDA_ASSERT(cudaMemcpy(this->buf_recv_h.data(), this->buf_send_d,
                                this->count1 * sizeof(TypeParam),
                                cudaMemcpyDeviceToHost));
        // mpi
        MPI_Bcast(this->buf_send_h.data(), this->count1, this->mpiDataType,
                  iroot, MPI_COMM_WORLD);
        // the result is in send buffer, so copy it to recv buffer for verify
        // step.
        this->buf_recv_mpi.assign(this->buf_send_h.begin(),
                                  this->buf_send_h.end());
        EXPECT_NO_FATAL_FAILURE(this->Verify(this->count1));
    }
}
