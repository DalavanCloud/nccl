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
TEST_F(mpi_test, test1) {
    // Initialize input values
    // 1...N
    for (int i = 0; i < count; i++) {
        buf_host[i] = TEST_ENV::mpi_rank + 1;
    }
    MCUDA_ASSERT(cudaMemcpy(buf_send, buf_host, count * sizeof(int),
                            cudaMemcpyHostToDevice));
    // Compute final value
    // SUM(1...N)
    const int ref = TEST_ENV::mpi_size * (TEST_ENV::mpi_size + 1) / 2;
    // Run allreduce
    for (int i = 0; i < 1; i++) {
        MNCCL_ASSERT(ncclAllReduce((const void*)buf_send, (void*)buf_recv,
                                   count, ncclInt, ncclSum, comm, stream));
    }
    // Check results
    MCUDA_ASSERT(cudaStreamSynchronize(stream));
    MCUDA_ASSERT(cudaMemcpy(buf_host, buf_recv, count * sizeof(int),
                            cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int v = 0; v < count; v++) {
        if (buf_host[v] != ref) {
            errors++;
            std::cerr << "[" << TEST_ENV::mpi_rank << "]"
                      << "Error at " << v << " : got " << buf_host[v]
                      << " instead of " << ref << std::endl;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INTEGER, MPI_SUM,
                  MPI_COMM_WORLD);
    MINT_EXPECT(0, errors);
}
