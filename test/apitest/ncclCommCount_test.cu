class ncclCommCount_test : public ::testing::Test {
  protected:
    ncclComm_t* comms = NULL;
    int* devList = NULL;
    int nVis = 0, ndev = 0;
    int count = -1;
    virtual void SetUp() {
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
        ndev = nVis;
        comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    };
    virtual void TearDown() {
        free(devList);
        if (comms != NULL) {
            for (int i = 0; i < nVis; ++i) {
                ncclCommDestroy(comms[i]);
            }
            free(comms);
        }
    };
};
TEST_F(ncclCommCount_test, basic) {
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, ndev, NULL));
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommCount(comms[i], &count));
        ASSERT_EQ(ndev, count);
    }
};
// 1.
TEST_F(ncclCommCount_test, comm_null) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommCount(NULL, &count));
};
// 2.
TEST_F(ncclCommCount_test, count_null) {
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, ndev, NULL));
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclInvalidArgument, ncclCommCount(comms[i], NULL));
    }
};
// 3.
TEST_F(ncclCommCount_test, comm_some) {
    ndev = nVis - (nVis > 1 ? 1 : 0);
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, ndev, NULL));
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommCount(comms[i], &count));
        ASSERT_EQ(ndev, count);
    };
}
