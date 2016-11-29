#include "ncclCommon_test.cuh"
template <typename DT>
class ncclAllGather_test : public ncclCommon_test<DT> {};
TYPED_TEST_CASE(ncclAllGather_test, testDataTypes);
// typical usage.
TYPED_TEST(ncclAllGather_test, basic) {
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "i" << i << ", "
                                                 << std::endl;
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(this->sendbuffs[i],
                                std::min(this->N, 1024 * 1024),
                                this->DataType(), this->recvbuffs[i],
                                this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
};
TYPED_TEST(ncclAllGather_test, DISABLED_pinned_mem) {
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "i" << i << ", "
                                                 << std::endl;
        EXPECT_EQ(ncclSuccess,
                  ncclAllGather(this->sendbuffs_pinned[i],
                                std::min(this->N, 1024 * 1024),
                                this->DataType(), this->recvbuffs_pinned[i],
                                this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
};
TYPED_TEST(ncclAllGather_test, host_mem) {
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "i" << i << ", "
                                                 << std::endl;
        EXPECT_EQ(ncclInvalidDevicePointer,
                  ncclAllGather(this->sendbuffs_host[i],
                                std::min(this->N, 1024 * 1024),
                                this->DataType(), this->recvbuffs_host[i],
                                this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
};
// sendbuff
TYPED_TEST(ncclAllGather_test, sendbuf_null) {
    int i = 0;
    EXPECT_EQ(ncclInvalidDevicePointer,
              ncclAllGather(NULL, std::min(this->N, 1024 * 1024),
                            this->DataType(), this->recvbuffs[i],
                            this->comms[i], this->streams[i]));
};
TYPED_TEST(ncclAllGather_test, sendbuf_wrong) {
    int i = 0, j = 1;
    ASSERT_EQ(cudaSuccess, cudaSetDevice(j));
    EXPECT_EQ(ncclInvalidDevicePointer,
              ncclAllGather(this->sendbuffs[i], std::min(this->N, 1024 * 1024),
                            this->DataType(), this->recvbuffs[i],
                            this->comms[i], this->streams[i]));
};
// recvbuff
TYPED_TEST(ncclAllGather_test, recvbuf_null) {
    int i = 0;
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    EXPECT_EQ(ncclInvalidDevicePointer,
              ncclAllGather(this->sendbuffs[i], std::min(this->N, 1024 * 1024),
                            this->DataType(), NULL, this->comms[i],
                            this->streams[i]));
}
// sendbuff and recvbuff not on the same device
TYPED_TEST(ncclAllGather_test, sendbuff_recvbuff_diff_device) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclAllGather(this->sendbuffs[i], std::min(this->N, 1024 * 1024),
                            this->DataType(), this->recvbuffs[j],
                            this->comms[i], this->streams[i]));
};
// N
TYPED_TEST(ncclAllGather_test, DISABLED_N_zero) {
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "i" << i << ", "
                                                 << std::endl;
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(this->sendbuffs[i], 0, this->DataType(),
                                this->recvbuffs[i], this->comms[i],
                                this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
};
TYPED_TEST(ncclAllGather_test, N_minus1) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllGather(this->sendbuffs[i], -1, this->DataType(),
                            this->recvbuffs[i], this->comms[i],
                            this->streams[i]));
};
// data type
TYPED_TEST(ncclAllGather_test, DataType_wrong) {
    int i = 0;
    ASSERT_EQ(ncclInvalidType,
              ncclAllGather(this->sendbuffs[i], std::min(this->N, 1024 * 1024),
                            nccl_NUM_TYPES, this->recvbuffs[i], this->comms[i],
                            this->streams[i]));
};
// comm
TYPED_TEST(ncclAllGather_test, comm_null) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllGather(this->sendbuffs[i], std::min(this->N, 1024 * 1024),
                            this->DataType(), this->recvbuffs[i], NULL,
                            this->streams[i]));
};
TYPED_TEST(ncclAllGather_test, comm_wrong) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclAllGather(this->sendbuffs[i], std::min(this->N, 1024 * 1024),
                            this->DataType(), this->recvbuffs[i],
                            this->comms[j], this->streams[i]));
};
// STREAM can be NULL.
// stream on a diff device
TYPED_TEST(ncclAllGather_test, stream_wrong) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclAllGather(this->sendbuffs[i], std::min(this->N, 1024 * 1024),
                            this->DataType(), this->recvbuffs[i],
                            this->comms[i], this->streams[j]));
};
// EOF
