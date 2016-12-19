#include "../include/macros.h"
#include "TEST_ENV.h"
#include "mpi.h"
#include "mpi_fixture.h"
ncclUniqueId mpi_test::commId = {};
ncclComm_t mpi_test::comm = NULL;
cudaStream_t mpi_test::stream = NULL;
int* mpi_test::buf_send = NULL;
int* mpi_test::buf_recv = NULL;
int* mpi_test::buf_host = NULL;
void mpi_test::SetUpTestCase() {
    // create NCCL Communicator
    NCCL_ASSERT(ncclGetUniqueId(&commId));
    MPI_Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, TEST_ENV::mpi_root,
              MPI_COMM_WORLD);
    NCCL_ASSERT(ncclCommInitRank(&comm, TEST_ENV::mpi_size, commId,
                                 TEST_ENV::mpi_rank));
    // create CUDA stream
    CUDA_ASSERT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // allocate data
    CUDA_ASSERT(cudaMalloc(&buf_send, count * sizeof(int)));
    CUDA_ASSERT(cudaMalloc(&buf_recv, count * sizeof(int)));
    buf_host = new int[count];
};
void mpi_test::TearDownTestCase() {
    CUDA_EXPECT(cudaFree(buf_send));
    buf_send = NULL;
    CUDA_EXPECT(cudaFree(buf_recv));
    buf_recv = NULL;
    delete[] buf_host;
    cudaStreamDestroy(stream);
    stream = NULL;
    ncclCommDestroy(comm);
    comm = NULL;
}
