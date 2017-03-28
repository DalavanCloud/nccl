#pragma once
#define CUDA_ASSERT(...)                                                       \
    do {                                                                       \
        cudaError_t err = __VA_ARGS__;                                         \
        ASSERT_EQ(cudaSuccess, err) << cudaGetErrorName(err) << ": "           \
                                    << cudaGetErrorString(err);                \
    } while (0);
#define NCCL_ASSERT(...)                                                       \
    do {                                                                       \
        ncclResult_t ret = __VA_ARGS__;                                        \
        ASSERT_EQ(ncclSuccess, ret) << ncclGetErrorString(ret);                \
    } while (0);
#define CUDA_EXPECT(...)                                                       \
    do {                                                                       \
        cudaError_t err = __VA_ARGS__;                                         \
        EXPECT_EQ(cudaSuccess, err) << cudaGetErrorName(err) << ": "           \
                                    << cudaGetErrorString(err);                \
    } while (0);
#define NCCL_EXPECT(...)                                                       \
    do {                                                                       \
        ncclResult_t ret = __VA_ARGS__;                                        \
        EXPECT_EQ(ncclSuccess, ret) << ncclGetErrorString(ret);                \
    } while (0);
//
#define L() std::cerr << __LINE__ << std::endl;
