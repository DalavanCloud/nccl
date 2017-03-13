#include "ncclCommon_test.cuh"
template <typename DT>
class ncclBcast_test : public ncclCommon_test<DT> {};
TYPED_TEST_CASE(ncclBcast_test, testDataTypes);
// typical usage.
TYPED_TEST(ncclBcast_test, basic) {
    for (int root = 0; root < this->nVis; ++root) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "root: " << root << ", "
                                                     << "i" << i << ", "
                                                     << std::endl;
            ASSERT_EQ(ncclSuccess,
                      ncclBcast(this->sendbuffs[i],
                                std::min(this->N, 32 * 1024), this->DataType(),
                                root, this->comms[i], this->streams[i]))
                << "root: " << root << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
};
TYPED_TEST(ncclBcast_test, host_mem) {
    for (int root = 0; root < this->nVis; ++root) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "root: " << root << ", "
                                                     << "i" << i << ", "
                                                     << std::endl;
            ASSERT_EQ(ncclInvalidDevicePointer,
                      ncclBcast(this->sendbuffs_host[i],
                                std::min(this->N, 32 * 1024), this->DataType(),
                                root, this->comms[i], this->streams[i]))
                << "root: " << root << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
};
TYPED_TEST(ncclBcast_test, pinned_mem) {
    for (int root = 0; root < this->nVis; ++root) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i)) << "root: " << root << ", "
                                                     << "i" << i << ", "
                                                     << std::endl;
            ASSERT_EQ(ncclSuccess,
                      ncclBcast(this->sendbuffs_pinned_device[i],
                                std::min(this->N, 32 * 1024), this->DataType(),
                                root, this->comms[i], this->streams[i]))
                << "root: " << root << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
};
// sendbuff
TYPED_TEST(ncclBcast_test, sendbuf_null) {
    int i = 0, root = 0;
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclBcast(NULL, std::min(this->N, 32 * 1024), this->DataType(),
                        root, this->comms[i], this->streams[i]));
};
TYPED_TEST(ncclBcast_test, sendbuf_wrong) {
    int i = 0, j = 1, root = 0;
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclBcast(this->sendbuffs[j], std::min(this->N, 32 * 1024),
                        this->DataType(), root, this->comms[i],
                        this->streams[i]));
};
// N
TYPED_TEST(ncclBcast_test, N_zero) {
   for (int root = 0; root < this->nVis; ++root) {
       ASSERT_EQ(ncclSuccess, ncclGroupStart());
       for (int i = 0; i < this->nVis; ++i) {
           ASSERT_EQ(cudaSuccess, cudaSetDevice(i))
               << "root: " << root << ", "
               << "i" << i << ", " << std::endl;
           ASSERT_EQ(ncclSuccess,
                     ncclBcast(this->sendbuffs[i], 0, this->DataType(),
                               root, this->comms[i], this->streams[i]))
               << "root: " << root << ", "
               << "i" << i << ", " << std::endl;
       }
       ASSERT_EQ(ncclSuccess, ncclGroupEnd());
   }
};
// data type
TYPED_TEST(ncclBcast_test, DataType_wrong) {
    int i = 0, root = 0;
    ASSERT_EQ(ncclInvalidType,
              ncclBcast(this->sendbuffs[i], std::min(this->N, 32 * 1024),
                        ncclNumTypes, root, this->comms[i],
                        this->streams[i]));
};
// root
TYPED_TEST(ncclBcast_test, root_minus1) {
    int i = 0, root = -1;
    ASSERT_EQ(ncclInvalidRank,
              ncclBcast(this->sendbuffs[i], std::min(this->N, 32 * 1024),
                        this->DataType(), root, this->comms[i],
                        this->streams[i]));
};
TYPED_TEST(ncclBcast_test, root_toobig) {
    int i = 0, root = 1000;
    ASSERT_EQ(ncclInvalidRank,
              ncclBcast(this->sendbuffs[i], std::min(this->N, 32 * 1024),
                        this->DataType(), root, this->comms[i],
                        this->streams[i]));
};
// comm
TYPED_TEST(ncclBcast_test, comm_null) {
    int i = 0, root = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclBcast(this->sendbuffs[i], std::min(this->N, 32 * 1024),
                        this->DataType(), root, NULL, this->streams[i]));
};
TYPED_TEST(ncclBcast_test, comm_wrong) {
    int i = 0, j = 1, root = 0;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclBcast(this->sendbuffs[i], std::min(this->N, 32 * 1024),
                        this->DataType(), root, this->comms[j],
                        this->streams[i]));
};
// STREAM can be NULL.
// stream on a diff device
TYPED_TEST(ncclBcast_test, DISABLED_stream_wrong) {
    int i = 0, j = 1, root = 0;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclBcast(this->sendbuffs[i], std::min(this->N, 32 * 1024),
                        this->DataType(), root, this->comms[i],
                        this->streams[j]));
};
