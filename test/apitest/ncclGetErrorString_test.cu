TEST(ncclGetErrorString, basic) {
    EXPECT_STREQ("no error", ncclGetErrorString(ncclSuccess));
    EXPECT_STREQ("unhandled cuda error", ncclGetErrorString(ncclUnhandledCudaError));
    EXPECT_STREQ("unhandled system error", ncclGetErrorString(ncclSystemError));
    EXPECT_STREQ("internal error", ncclGetErrorString(ncclInternalError));
    EXPECT_STREQ("invalid argument", ncclGetErrorString(ncclInvalidArgument));
    EXPECT_STREQ("invalid usage", ncclGetErrorString(ncclInvalidUsage));
    EXPECT_STREQ("unknown result code", ncclGetErrorString((ncclResult_t)-1));
};
