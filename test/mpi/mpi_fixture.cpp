#include "mpi.h"
#include "../include/macros.h"
#include "ErrorChecker.h"
#include "TEST_ENV.h"
#include "mpi_fixture.h"
template <typename T>
const int mpi_test<T>::count1 = 32;
template <typename T>
int mpi_test<T>::countN = 0;
template <typename T>
ncclUniqueId mpi_test<T>::commId = {};
template <typename T>
ncclComm_t mpi_test<T>::comm = NULL;
template <typename T>
cudaStream_t mpi_test<T>::stream = NULL;
template <typename T>
T* mpi_test<T>::buf_send_d = NULL;
template <typename T>
T* mpi_test<T>::buf_recv_d = NULL;
template <typename T>
std::vector<T> mpi_test<T>::buf_send_h;
template <typename T>
std::vector<T> mpi_test<T>::buf_recv_h;
template <typename T>
std::vector<T> mpi_test<T>::buf_recv_mpi;
template <typename T>
const std::vector<ncclRedOp_t> mpi_test<T>::RedOps = {ncclSum, ncclProd,
                                                      ncclMax, ncclMin};
template <typename T>
const std::map<ncclRedOp_t, MPI_Op> mpi_test<T>::MpiOps = {{ncclSum, MPI_SUM},
                                                           {ncclProd, MPI_PROD},
                                                           {ncclMax, MPI_MAX},
                                                           {ncclMin, MPI_MIN}};
template <typename T>
void mpi_test<T>::SetUpTestCase() {
    countN = count1 * TEST_ENV::mpi_size;
    // create NCCL Communicator
    MNCCL_ASSERT(ncclGetUniqueId(&commId));
    MPI_Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, TEST_ENV::mpi_root,
              MPI_COMM_WORLD);
    MNCCL_ASSERT(ncclCommInitRank(&comm, TEST_ENV::mpi_size, commId,
                                  TEST_ENV::mpi_rank));
    AllocData();
};
template <typename T>
void mpi_test<T>::TearDownTestCase() {
    FreeData();
}
template <typename T>
void mpi_test<T>::FreeData() {
    MCUDA_EXPECT(cudaFree(buf_send_d));
    buf_send_d = NULL;
    MCUDA_EXPECT(cudaFree(buf_recv_d));
    buf_recv_d = NULL;
    cudaStreamDestroy(stream);
    stream = NULL;
    ncclCommDestroy(comm);
    comm = NULL;
}
template <typename T>
void mpi_test<T>::AllocData() {
    // stream
    MCUDA_ASSERT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // device memory
    MCUDA_ASSERT(cudaMalloc(&buf_send_d, count1 * sizeof(T)));
    MCUDA_ASSERT(cudaMalloc(&buf_recv_d, countN * sizeof(T)));
    // host memory
    buf_send_h.resize(count1);
    buf_recv_h.resize(countN);
    buf_recv_mpi.resize(countN);
}
template <typename T>
void mpi_test<T>::SetUp() {
    this->InitInput();
}
template <typename T>
void mpi_test<T>::TearDown() {
    this->buf_send_h.assign(this->buf_send_h.size(), 0);
    this->buf_recv_h.assign(this->buf_recv_h.size(), 0);
    this->buf_recv_mpi.assign(this->buf_recv_mpi.size(), 0);
}
template <typename T>
void mpi_test<T>::InitInput() {
    // Initialize input values  // rank * count + i
    for (int i = 0; i < this->count1; i++) {
        this->buf_send_h[i] = TEST_ENV::mpi_rank * this->count1 + i;
    }
    MCUDA_ASSERT(cudaMemcpy(this->buf_send_d, this->buf_send_h.data(),
                            this->count1 * sizeof(T), cudaMemcpyHostToDevice));
    MPI_Barrier(MPI_COMM_WORLD);
}
template <typename T>
void mpi_test<T>::Verify(const int length) const {
    MPI_Barrier(MPI_COMM_WORLD);
    int errors = 0;
    for (int i = 0; i < length; ++i) {
        if (this->buf_recv_h[i] != this->buf_recv_mpi[i]) {
            ++errors;
        }
    }
    MINT_EXPECT(0, errors);
}
#define GEN_DT(X, Y, Z)                                                        \
    template <>                                                                \
    const ncclDataType_t mpi_test<X>::ncclDataType = Y;                        \
    template <>                                                                \
    const MPI_Datatype mpi_test<X>::mpiDataType = Z;
GEN_DT(char, ncclChar, MPI_CHAR);
GEN_DT(int, ncclInt, MPI_INT);
GEN_DT(float, ncclFloat, MPI_FLOAT);
GEN_DT(double, ncclDouble, MPI_DOUBLE);
GEN_DT(long long, ncclInt64, MPI_LONG_LONG);
GEN_DT(unsigned long long, ncclUint64, MPI_UNSIGNED_LONG_LONG);
#undef GEN_DT
#define INST(X) template class mpi_test<X>;
INST(char);
INST(int);
INST(float);
INST(double);
INST(long long);
INST(unsigned long long);
#undef INST
