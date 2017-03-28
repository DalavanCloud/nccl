class ncclCommInitRank_test : public ::testing::Test {
  protected:
    ncclComm_t comm = NULL;
    int ndev = 1;
    ncclUniqueId commId;
    int rank = 0;
    virtual void SetUp() {
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
    };
    virtual void TearDown() {
        if (comm == NULL) {
          // This is needed to free the Unique Id
          ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, 1, commId, 0));
        }
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
    };
};
TEST_F(ncclCommInitRank_test, basic) {
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, ndev, commId, rank))
        << "This test should be passed always.";
}
TEST_F(ncclCommInitRank_test, comm_null) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(NULL, ndev, commId, rank));
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, ndev, commId, rank));
}
#if 0 // don't test this.
TEST_F(ncclCommInitRank_test, commId_uninitialized) {
    ncclUniqueId id;
    ASSERT_NE(ncclSuccess, ncclCommInitRank(&comm, ndev, id, rank))
        << "should an uninitialized unique id be used?";
}
#endif
TEST_F(ncclCommInitRank_test, ndev_zero) {
    ASSERT_EQ(ncclInvalidArgument,
              ncclCommInitRank(&comm, 0, commId, rank));
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, ndev, commId, rank));
}
TEST_F(ncclCommInitRank_test, dev_negative) {
    ASSERT_EQ(ncclInvalidArgument,
              ncclCommInitRank(&comm, -1, commId, rank));
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, ndev, commId, rank));
}
TEST_F(ncclCommInitRank_test, rank_outofboundary) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(&comm, ndev, commId, 1));
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, ndev, commId, rank));
}
TEST_F(ncclCommInitRank_test, rank_negative) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(&comm, ndev, commId, -1));
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, ndev, commId, rank));
}
TEST_F(ncclCommInitRank_test, DISABLED_dev_too_many) { // cause dead loop
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(&comm, 10, commId, rank));
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, ndev, commId, rank));
}
