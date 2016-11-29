class ncclCommUserRank_test : public ::testing::Test {
  protected:
    ncclComm_t* comms = NULL;
    int nVis = 0, rank = -1;
    virtual void SetUp() {
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
        comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
        ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
    };
    virtual void TearDown() {
        if (comms != NULL) {
            for (int i = 0; i < nVis; ++i) {
                ncclCommDestroy(comms[i]);
            }
            free(comms);
        }
    };
};
TEST_F(ncclCommUserRank_test, basic) {
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommUserRank(comms[i], &rank));
        ASSERT_EQ(rank, i);
    }
}
TEST_F(ncclCommUserRank_test, null_comm) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommUserRank(NULL, &rank));
}
TEST_F(ncclCommUserRank_test, null_rank) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommUserRank(comms[0], NULL));
}
// EOF
