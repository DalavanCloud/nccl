TEST(ncclCommDestroy, basic) {
    int ndev = 1;
    ncclComm_t* comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, ndev, NULL));
    for (int i = 0; i < ndev; ++i)
        ncclCommDestroy(comms[i]);
    SUCCEED();
}
TEST(ncclCommDestroy, null) {
    ncclCommDestroy(NULL);
    SUCCEED();
}
