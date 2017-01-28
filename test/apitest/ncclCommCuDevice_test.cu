class ncclCommCuDevice_test : public ::testing::Test {
  protected:
    ncclComm_t* comms = NULL;
    int nVis = 0, device = -1;
    virtual void SetUp() {
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
        comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
        ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
    };
    virtual void TearDown() {
        if (comms != NULL) {
            for (int i = 0; i < nVis; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
            }
            free(comms);
        }
    };
};
TEST_F(ncclCommCuDevice_test, basic) {
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommCuDevice(comms[i], &device));
        ASSERT_EQ(device, i);
    }
}
TEST_F(ncclCommCuDevice_test, null_comm) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommCuDevice(NULL, &device));
}
TEST_F(ncclCommCuDevice_test, null_device) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommCuDevice(comms[0], NULL));
}
// EOF
