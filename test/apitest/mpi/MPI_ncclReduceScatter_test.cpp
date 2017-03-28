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
TYPED_TEST(mpi_test, ncclReduceScatter_basic) {
    this->loop_factor = this->RedOps.size();
    PERF_BEGIN();
    for (auto op : this->RedOps) {
        MPI_Barrier(MPI_COMM_WORLD);
        // run nccl
        MNCCL_ASSERT(ncclReduceScatter(
            (const void*)this->buf_send_d, (void*)this->buf_recv_d,
            this->count1, this->ncclDataType, op, this->comm, this->stream));
        MCUDA_ASSERT(cudaStreamSynchronize(this->stream));
        if (!isPerf) {
            MCUDA_ASSERT(cudaMemcpy(this->buf_recv_h.data(), this->buf_recv_d,
                                    this->count1 * sizeof(TypeParam),
                                    cudaMemcpyDeviceToHost));
            // mpi result
            std::vector<int> recvcounts(TEST_ENV::mpi_size, this->count1);
            MPI_Reduce_scatter(this->buf_send_h.data(),
                               this->buf_recv_mpi.data(), recvcounts.data(),
                               this->mpiDataType, this->MpiOps.at(op),
                               MPI_COMM_WORLD);
            EXPECT_NO_FATAL_FAILURE(this->Verify(this->count1));
        }
    }
    PERF_END();
}
