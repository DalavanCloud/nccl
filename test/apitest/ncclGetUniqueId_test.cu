TEST(ncclGetUniqueId, basic) {
    ncclUniqueId id;
    EXPECT_EQ(ncclSuccess, ncclGetUniqueId(&id));
}
TEST(ncclGetUniqueId, null) {
    EXPECT_EQ(ncclInvalidArgument, ncclGetUniqueId(NULL));
}
