#include "ncclCommon_test.cuh"
template <typename DT>
class ncclReduceScatter_test : public ncclCommon_test<DT> {};
TYPED_TEST_CASE(ncclReduceScatter_test, testDataTypes);
// typical usage.
TYPED_TEST(ncclReduceScatter_test, basic) {
    for (ncclRedOp_t op : this->RedOps) {
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "op: " << op << ", "
                                                     << "i" << i << ", "
                                                     << std::endl;
            ASSERT_EQ(ncclSuccess,
                      ncclReduceScatter(this->sendbuffs[i], this->recvbuffs[i],
                                        std::min(this->N/this->nVis, 1024 * 1024),
                                        this->DataType(), op, this->comms[i],
                                        this->streams[i]))
                << "op: " << op << ", "
                << "i" << i << ", " << std::endl;
        }
    }
};
TYPED_TEST(ncclReduceScatter_test, host_mem) {
    for (ncclRedOp_t op : this->RedOps) {
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "op: " << op << ", "
                                                     << "i" << i << ", "
                                                     << std::endl;
            ASSERT_EQ(ncclInvalidDevicePointer,
                      ncclReduceScatter(
                          this->sendbuffs_host[i], this->recvbuffs_host[i],
                          std::min(this->N/this->nVis, 1024 * 1024), this->DataType(), op,
                          this->comms[i], this->streams[i]))
                << "op: " << op << ", "
                << "i" << i << ", " << std::endl;
        }
    }
};
TYPED_TEST(ncclReduceScatter_test, DISABLED_pinned_mem) {
    for (ncclRedOp_t op : this->RedOps) {
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "op: " << op << ", "
                                                     << "i" << i << ", "
                                                     << std::endl;
            ASSERT_EQ(ncclSuccess,
                      ncclReduceScatter(
                          this->sendbuffs_pinned[i], this->recvbuffs_pinned[i],
                          std::min(this->N/this->nVis, 1024 * 1024), this->DataType(), op,
                          this->comms[i], this->streams[i]))
                << "op: " << op << ", "
                << "i" << i << ", " << std::endl;
        }
    }
};
// sendbuff
TYPED_TEST(ncclReduceScatter_test, sendbuf_null) {
    int i = 0;
    EXPECT_EQ(ncclInvalidDevicePointer,
              ncclReduceScatter(NULL, this->recvbuffs[i],
                                std::min(this->N/this->nVis, 1024 * 1024),
                                this->DataType(), this->RedOps[0],
                                this->comms[i], this->streams[i]));
};
// recvbuff
TYPED_TEST(ncclReduceScatter_test, recvbuf_null) {
    int i = 0;
    EXPECT_EQ(ncclInvalidDevicePointer,
              ncclReduceScatter(this->sendbuffs[i], NULL,
                                std::min(this->N/this->nVis, 1024 * 1024),
                                this->DataType(), this->RedOps[0],
                                this->comms[i], this->streams[i]));
};
// sendbuff and recvbuff not on the same device
TYPED_TEST(ncclReduceScatter_test, sendbuff_recvbuff_diff_device) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclReduceScatter(this->sendbuffs[i], this->recvbuffs[j],
                                std::min(this->N/this->nVis, 1024 * 1024),
                                this->DataType(), this->RedOps[0],
                                this->comms[i], this->streams[i]));
};
// N
TYPED_TEST(ncclReduceScatter_test, N_zero) {
    for (ncclRedOp_t op : this->RedOps) {
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "op: " << op << ", "
                                                     << "i" << i << ", "
                                                     << std::endl;
            ASSERT_EQ(ncclSuccess,
                      ncclReduceScatter(this->sendbuffs[i], this->recvbuffs[i],
                                        0, this->DataType(), this->RedOps[0],
                                        this->comms[i], this->streams[i]))
                << "op: " << op << ", "
                << "i" << i << ", " << std::endl;
        }
    }
};
// data type
TYPED_TEST(ncclReduceScatter_test, DataType_wrong) {
    int i = 0;
    ASSERT_EQ(ncclInvalidType,
              ncclReduceScatter(this->sendbuffs[i], this->recvbuffs[i],
                                std::min(this->N/this->nVis, 1024 * 1024), ncclNumTypes,
                                this->RedOps[0], this->comms[i],
                                this->streams[i]));
};
// op
TYPED_TEST(ncclReduceScatter_test, op_wrong) {
    int i = 0;
    ASSERT_EQ(ncclInvalidOperation,
              ncclReduceScatter(this->sendbuffs[i], this->recvbuffs[i],
                                std::min(this->N/this->nVis, 1024 * 1024),
                                this->DataType(), ncclNumOps, this->comms[i],
                                this->streams[i]));
};
// comm
TYPED_TEST(ncclReduceScatter_test, comm_null) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclReduceScatter(this->sendbuffs[i], this->recvbuffs[i],
                                std::min(this->N/this->nVis, 1024 * 1024),
                                this->DataType(), this->RedOps[0], NULL,
                                this->streams[i]));
};
TYPED_TEST(ncclReduceScatter_test, comm_wrong) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclReduceScatter(this->sendbuffs[i], this->recvbuffs[i],
                                std::min(this->N/this->nVis, 1024 * 1024),
                                this->DataType(), this->RedOps[0],
                                this->comms[j], this->streams[i]));
};
// STREAM can be NULL.
// stream on a diff device
TYPED_TEST(ncclReduceScatter_test, DISABLED_stream_wrong) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclReduceScatter(this->sendbuffs[i], this->recvbuffs[i],
                                std::min(this->N/this->nVis, 1024 * 1024),
                                this->DataType(), this->RedOps[0],
                                this->comms[i], this->streams[j]));
};
// EOF
