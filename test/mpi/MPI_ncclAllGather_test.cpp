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
TYPED_TEST(mpi_test, ncclAllGather_basic) {
    PERF_BEGIN();
    MPI_Barrier(MPI_COMM_WORLD);
    MNCCL_ASSERT(ncclAllGather(
        (const void*)this->buf_send_d, (void*)this->buf_recv_d,
        this->count1, this->ncclDataType, this->comm, this->stream));
    MCUDA_ASSERT(cudaStreamSynchronize(this->stream));
    if (!isPerf) {
        MCUDA_ASSERT(cudaMemcpy(this->buf_recv_h.data(), this->buf_recv_d,
                                this->countN * sizeof(TypeParam),
                                cudaMemcpyDeviceToHost));
        MPI_Allgather(this->buf_send_h.data(), this->count1, this->mpiDataType,
                      this->buf_recv_mpi.data(), this->count1,
                      this->mpiDataType, MPI_COMM_WORLD);
        EXPECT_NO_FATAL_FAILURE(this->Verify(this->countN));
    }
    PERF_END();
}
