#include "TEST_ENV.h"
#include <mpi.h>
#include <vector>
#include "ErrorChecker.h"
template <typename T>
void expect_log(const int rank, const T expected, const T code);
template <typename T>
void expect_log(const int rank, const T expected, const T code) {
    EXPECT_EQ(expected, code) << "rank: " << rank;
}
template void expect_log(const int rank, const int expected, const int code);
template <>
void expect_log(const int rank, const cudaError_t expected,
                const cudaError_t code) {
    EXPECT_EQ(expected, code) << "rank: " << rank
                              << ". err: " << cudaGetErrorName(code) << ": "
                              << cudaGetErrorString(code);
}
template <>
void expect_log(const int rank, const ncclResult_t expected,
                const ncclResult_t code) {
    EXPECT_EQ(expected, code) << "rank: " << rank
                              << ". err: " << ncclGetErrorString(code);
}
template <typename T>
void ErrorChecker(const T expected, const T code) {
    std::vector<T> retcodes(TEST_ENV::mpi_size);
    bool ret = true;
    MPI_Allgather(&code, sizeof(T), MPI_CHAR, retcodes.data(), sizeof(T),
                  MPI_CHAR, MPI_COMM_WORLD);
    for (unsigned int i = 0; i < retcodes.size(); ++i) {
        expect_log<T>(i, expected, retcodes.at(i));
        if (expected != retcodes.at(i)) {
            ret = false;
        }
    }
    ASSERT_EQ(true, ret);
}
template void ErrorChecker(const cudaError_t expected, const cudaError_t code);
template void ErrorChecker(const ncclResult_t expected,
                           const ncclResult_t code);
template void ErrorChecker(const int expected, const int code);
