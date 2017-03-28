TEST(ncclGetUniqueId, basic) {
    ncclUniqueId id;
    EXPECT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    // Free resources
    ncclComm_t comm;
    EXPECT_EQ(ncclSuccess, ncclCommInitRank(&comm, 1, id, 0));
    EXPECT_EQ(ncclSuccess, ncclCommDestroy(comm));
}
TEST(ncclGetUniqueId, null) {
    EXPECT_EQ(ncclInvalidArgument, ncclGetUniqueId(NULL));
}
