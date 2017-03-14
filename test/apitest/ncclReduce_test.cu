#include "ncclCommon_test.cuh"
template <typename DT>
class ncclReduce_test : public ncclCommon_test<DT> {};
TYPED_TEST_CASE(ncclReduce_test, testDataTypes);
// typical usage.
TYPED_TEST(ncclReduce_test, basic) {
    for (ncclRedOp_t op : this->RedOps) {
        for (int root = 0; root < this->nVis; ++root) {
            ASSERT_EQ(ncclSuccess, ncclGroupStart());
            for (int i = 0; i < this->nVis; ++i) {
                ASSERT_EQ(cudaSuccess, cudaSetDevice(i))
                    << "op: " << op << ", "
                    << "root: " << root << ", "
                    << "i" << i << ", " << std::endl;
                ASSERT_EQ(ncclSuccess,
                          ncclReduce(this->sendbuffs[i], i == root ? this->recvbuffs[i] : NULL,
                                     std::min(this->N, 1024 * 1024),
                                     this->DataType(), op, root, this->comms[i],
                                     this->streams[i]))
                    << "op: " << op << ", "
                    << "root: " << root << ", "
                    << "i" << i << ", " << std::endl;
            }
            ASSERT_EQ(ncclSuccess, ncclGroupEnd());
        }
    }
};
TYPED_TEST(ncclReduce_test, host_mem) {
    for (ncclRedOp_t op : this->RedOps) {
        for (int root = 0; root < this->nVis; ++root) {
            ASSERT_EQ(ncclSuccess, ncclGroupStart());
            for (int i = 0; i < this->nVis; ++i) {
                ASSERT_EQ(cudaSuccess, cudaSetDevice(i))
                    << "op: " << op << ", "
                    << "root: " << root << ", "
                    << "i" << i << ", " << std::endl;
                ASSERT_EQ(
                    ncclInvalidDevicePointer,
                    ncclReduce(this->sendbuffs_host[i], this->recvbuffs_host[i],
                               std::min(this->N, 1024 * 1024), this->DataType(),
                               op, root, this->comms[i], this->streams[i]))
                    << "op: " << op << ", "
                    << "root: " << root << ", "
                    << "i" << i << ", " << std::endl;
            }
            ASSERT_EQ(ncclInvalidDevicePointer, ncclGroupEnd());
        }
    }
};
TYPED_TEST(ncclReduce_test, pinned_mem) {
    for (ncclRedOp_t op : this->RedOps) {
        for (int root = 0; root < this->nVis; ++root) {
            ASSERT_EQ(ncclSuccess, ncclGroupStart());
            for (int i = 0; i < this->nVis; ++i) {
                ASSERT_EQ(cudaSuccess, cudaSetDevice(i))
                    << "op: " << op << ", "
                    << "root: " << root << ", "
                    << "i" << i << ", " << std::endl;
                ASSERT_EQ(ncclSuccess,
                          ncclReduce(this->sendbuffs_pinned_device[i],
                                     this->recvbuffs_pinned_device[i],
                                     std::min(this->N, 1024 * 1024),
                                     this->DataType(), op, root, this->comms[i],
                                     this->streams[i]))
                    << "op: " << op << ", "
                    << "root: " << root << ", "
                    << "i" << i << ", " << std::endl;
            }
            ASSERT_EQ(ncclSuccess, ncclGroupEnd());
        }
    }
};
// sendbuff
TYPED_TEST(ncclReduce_test, sendbuf_null) {
    int i = 0, root = 0;
    EXPECT_EQ(ncclInvalidDevicePointer,
              ncclReduce(NULL, this->recvbuffs[i],
                         std::min(this->N, 1024 * 1024), this->DataType(),
                         this->RedOps[0], root, this->comms[i],
                         this->streams[i]));
};
// recvbuff
// root can't be null
// non root can be null
TYPED_TEST(ncclReduce_test, recvbuf_root_null) {
    for (ncclRedOp_t op : this->RedOps) {
        for (int root = 0; root < this->nVis; ++root) {
            ASSERT_EQ(ncclSuccess, ncclGroupStart());
            for (int i = 0; i < this->nVis; ++i) {
                ASSERT_EQ(cudaSuccess, cudaSetDevice(i))
                    << "op: " << op << ", "
                    << "root: " << root << ", "
                    << "i" << i << ", " << std::endl;
                ASSERT_EQ(root != i ? ncclSuccess : ncclInvalidDevicePointer,
                          ncclReduce(this->sendbuffs[i], NULL,
                                     std::min(this->N, 1024 * 1024),
                                     this->DataType(), this->RedOps[0], root,
                                     this->comms[i], this->streams[i]))
                    << "op: " << op << ", "
                    << "root: " << root << ", "
                    << "i" << i << ", " << std::endl;
            }
            ASSERT_EQ(ncclInvalidDevicePointer, ncclGroupEnd());
        }
    }
};
TYPED_TEST(ncclReduce_test, recvbuff_nonroot_null) {
    for (ncclRedOp_t op : this->RedOps) {
        for (int root = 0; root < this->nVis; ++root) {
            ASSERT_EQ(ncclSuccess, ncclGroupStart());
            for (int i = 0; i < this->nVis; ++i) {
                ASSERT_EQ(cudaSuccess, cudaSetDevice(i))
                    << "op: " << op << ", "
                    << "root: " << root << ", "
                    << "i" << i << ", " << std::endl;
                ASSERT_EQ(ncclSuccess,
                          ncclReduce(this->sendbuffs[i],
                                     root == i ? this->recvbuffs[i] : NULL,
                                     std::min(this->N, 1024 * 1024),
                                     this->DataType(), this->RedOps[0], root,
                                     this->comms[i], this->streams[i]))
                    << "op: " << op << ", "
                    << "root: " << root << ", "
                    << "i" << i << ", " << std::endl;
            }
            ASSERT_EQ(ncclSuccess, ncclGroupEnd());
        }
    }
};
// root device, sendbuff and recvbuff not on the same device
TYPED_TEST(ncclReduce_test, root_sendbuff_recvbuff_diff_device) {
    int i = 0, j = 1, root = 0;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclReduce(this->sendbuffs[i], this->recvbuffs[j],
                         std::min(this->N, 1024 * 1024), this->DataType(),
                         this->RedOps[0], root, this->comms[i],
                         this->streams[i]));
};
// N
TYPED_TEST(ncclReduce_test, N_zero) {
    for (ncclRedOp_t op : this->RedOps) {
        for (int root = 0; root < this->nVis; ++root) {
            ASSERT_EQ(ncclSuccess, ncclGroupStart());
            for (int i = 0; i < this->nVis; ++i) {
                ASSERT_EQ(cudaSuccess, cudaSetDevice(i))
                    << "op: " << op << ", "
                    << "root: " << root << ", "
                    << "i" << i << ", " << std::endl;
                ASSERT_EQ(ncclSuccess,
                          ncclReduce(this->sendbuffs[i], this->recvbuffs[i], 0,
                                     this->DataType(), this->RedOps[0], root,
                                     this->comms[i], this->streams[i]))
                    << "op: " << op << ", "
                    << "root: " << root << ", "
                    << "i" << i << ", " << std::endl;
            }
            ASSERT_EQ(ncclSuccess, ncclGroupEnd());
        }
    }
};
// data type
TYPED_TEST(ncclReduce_test, DataType_wrong) {
    int i = 0, root = 0;
    ASSERT_EQ(ncclInvalidType,
              ncclReduce(this->sendbuffs[i], this->recvbuffs[i],
                         std::min(this->N, 1024 * 1024), ncclNumTypes,
                         this->RedOps[0], root, this->comms[i],
                         this->streams[i]));
};
// op
TYPED_TEST(ncclReduce_test, op_wrong) {
    int i = 0, root = 0;
    ASSERT_EQ(ncclInvalidOperation,
              ncclReduce(this->sendbuffs[i], this->recvbuffs[i],
                         std::min(this->N, 1024 * 1024), this->DataType(),
                         ncclNumOps, root, this->comms[i], this->streams[i]));
};
// root
TYPED_TEST(ncclReduce_test, root_minus1) {
    int i = 0, root = -1;
    ASSERT_EQ(ncclInvalidRank,
              ncclReduce(this->sendbuffs[i], this->recvbuffs[i],
                         std::min(this->N, 1024 * 1024), this->DataType(),
                         this->RedOps[0], root, this->comms[i],
                         this->streams[i]));
};
TYPED_TEST(ncclReduce_test, root_toobig) {
    int i = 0, root = 1000;
    ASSERT_EQ(ncclInvalidRank,
              ncclReduce(this->sendbuffs[i], this->recvbuffs[i],
                         std::min(this->N, 1024 * 1024), this->DataType(),
                         this->RedOps[0], root, this->comms[i],
                         this->streams[i]));
};
// comm
TYPED_TEST(ncclReduce_test, comm_null) {
    int i = 0, root = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclReduce(this->sendbuffs[i], this->recvbuffs[i],
                         std::min(this->N, 1024 * 1024), this->DataType(),
                         this->RedOps[0], root, NULL, this->streams[i]));
};
TYPED_TEST(ncclReduce_test, comm_wrong) {
    int i = 0, j = 1, root = 0;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclReduce(this->sendbuffs[i], this->recvbuffs[i],
                         std::min(this->N, 1024 * 1024), this->DataType(),
                         this->RedOps[0], root, this->comms[j],
                         this->streams[i]));
};
// STREAM can be NULL.
// stream on a diff device
#if 0 // nccl can't handle this.
TYPED_TEST(ncclReduce_test, stream_wrong) {
    int i = 0, j = 1, root = 0;
    ASSERT_EQ(ncclInvalidDevicePointer,
              ncclReduce(this->sendbuffs[i], this->recvbuffs[i],
                         std::min(this->N, 1024 * 1024), this->DataType(),
                         this->RedOps[0], root, this->comms[i],
                         this->streams[j]));
};
#endif
