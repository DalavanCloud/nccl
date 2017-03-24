#pragma once
/// TODO: can't be compiled with T=bool
// all processes should check the return code.
// if only the root rank process assert, then the other process will hang.
template <typename T>
void ErrorChecker(const T expected, const T code);
#define MCUDA_ASSERT(...)                                                      \
    do {                                                                       \
        ASSERT_NO_FATAL_FAILURE(                                               \
            ErrorChecker<cudaError_t>(cudaSuccess, __VA_ARGS__));              \
    } while (0);
#define MCUDA_EXPECT(...)                                                      \
    do {                                                                       \
        EXPECT_NO_FATAL_FAILURE(                                               \
            ErrorChecker<cudaError_t>(cudaSuccess, __VA_ARGS__));              \
    } while (0);
#define MNCCL_ASSERT(...)                                                      \
    do {                                                                       \
        ASSERT_NO_FATAL_FAILURE(                                               \
            ErrorChecker<ncclResult_t>(ncclSuccess, __VA_ARGS__));             \
    } while (0);
#define MNCCL_EXPECT(...)                                                      \
    do {                                                                       \
        EXPECT_NO_FATAL_FAILURE(                                               \
            ErrorChecker<ncclResult_t>(ncclSuccess, __VA_ARGS__));             \
    } while (0);
#define MINT_EXPECT(M, N)                                                      \
    do {                                                                       \
        EXPECT_NO_FATAL_FAILURE(ErrorChecker<int>(M, N));                      \
    } while (0);
#define MINT_ASSERT(M, N)                                                      \
    do {                                                                       \
        ASSERT_NO_FATAL_FAILURE(ErrorChecker<int>(M, N));                      \
    } while (0);
//// EOF ////
