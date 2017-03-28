TEST(ncclCommDestroy, basic) {
    int ndev = 1;
    ncclComm_t* comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, ndev, NULL));
    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
    SUCCEED();
}
TEST(ncclCommDestroy, null) {
    ASSERT_EQ(ncclSuccess, ncclCommDestroy(NULL));
    SUCCEED();
}
