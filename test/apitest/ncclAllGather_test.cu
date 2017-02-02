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
                  ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                                std::min(this->N/this->nVis, 1024 * 1024),
                                this->DataType(), this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
};
TYPED_TEST(ncclAllGather_test, host_mem) {
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "i" << i << ", "
                                                 << std::endl;
        EXPECT_EQ(ncclInvalidDevicePointer,
                  ncclAllGather(this->sendbuffs_host[i], this->recvbuffs_host[i],
                                std::min(this->N/this->nVis, 1024 * 1024),
                                this->DataType(), this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
};
TYPED_TEST(ncclAllGather_test, DISABLED_pinned_mem) {
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "i" << i << ", "
                                                 << std::endl;
        EXPECT_EQ(ncclSuccess,
                  ncclAllGather(this->sendbuffs_pinned[i], this->recvbuffs_pinned[i],
                                std::min(this->N/this->nVis, 1024 * 1024),
                                this->DataType(), this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
};
// sendbuff
TYPED_TEST(ncclAllGather_test, sendbuf_null) {
    int i = 0;
    EXPECT_EQ(ncclInvalidDevicePointer,
              ncclAllGather(NULL, this->recvbuffs[i],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(), this->comms[i], this->streams[i]));
};
TYPED_TEST(ncclAllGather_test, sendbuf_wrong) {
    int i = 0, j = 1;
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    EXPECT_EQ(ncclInvalidDevicePointer,
              ncclAllGather(this->sendbuffs[j], this->recvbuffs[i],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(),
                            this->comms[i], this->streams[i]));
};
// recvbuff
TYPED_TEST(ncclAllGather_test, recvbuf_null) {
    int i = 0;
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    EXPECT_EQ(ncclInvalidDevicePointer,
              ncclAllGather(this->sendbuffs[i], NULL,
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(), this->comms[i], this->streams[i]));
}
// sendbuff and recvbuff not on the same device
TYPED_TEST(ncclAllGather_test, sendbuff_recvbuff_diff_device) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclAllGather(this->sendbuffs[i], this->recvbuffs[j],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(), this->comms[i], this->streams[i]));
};
// N
TYPED_TEST(ncclAllGather_test, DISABLED_N_zero) {
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "i" << i << ", "
                                                 << std::endl;
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(this->sendbuffs[i], this->recvbuffs[i], 0,
                                this->DataType(), this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
};
// data type
TYPED_TEST(ncclAllGather_test, DataType_wrong) {
    int i = 0;
    ASSERT_EQ(ncclInvalidType,
              ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            nccl_NUM_TYPES, this->comms[i], this->streams[i]));
};
// comm
TYPED_TEST(ncclAllGather_test, comm_null) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(), NULL, this->streams[i]));
};
TYPED_TEST(ncclAllGather_test, comm_wrong) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(), this->comms[j], this->streams[i]));
};
// STREAM can be NULL.
// stream on a diff device
TYPED_TEST(ncclAllGather_test, DISABLED_stream_wrong) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(), this->comms[i], this->streams[j]));
};
// EOF
