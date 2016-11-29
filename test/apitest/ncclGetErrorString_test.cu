TEST(ncclGetErrorString, basic) {
    EXPECT_STREQ("no error", ncclGetErrorString(ncclSuccess));
    EXPECT_STREQ("unhandled cuda error", ncclGetErrorString(ncclUnhandledCudaError));
    EXPECT_STREQ("system error", ncclGetErrorString(ncclSystemError));
    EXPECT_STREQ("internal error", ncclGetErrorString(ncclInternalError));
    EXPECT_STREQ("invalid device pointer", ncclGetErrorString(ncclInvalidDevicePointer));
    EXPECT_STREQ("invalid rank", ncclGetErrorString(ncclInvalidRank));
    EXPECT_STREQ("unsupported device count", ncclGetErrorString(ncclUnsupportedDeviceCount));
    EXPECT_STREQ("device not found", ncclGetErrorString(ncclDeviceNotFound));
    EXPECT_STREQ("invalid device index", ncclGetErrorString(ncclInvalidDeviceIndex));
    EXPECT_STREQ("lib wrapper not initialized", ncclGetErrorString(ncclLibWrapperNotSet));
    EXPECT_STREQ("cuda malloc failed", ncclGetErrorString(ncclCudaMallocFailed));
    EXPECT_STREQ("parameter mismatch between ranks", ncclGetErrorString(ncclRankMismatch));
    EXPECT_STREQ("invalid argument", ncclGetErrorString(ncclInvalidArgument));
    EXPECT_STREQ("invalid data type", ncclGetErrorString(ncclInvalidType));
    EXPECT_STREQ("invalid reduction operations", ncclGetErrorString(ncclInvalidOperation));
    EXPECT_STREQ("unknown result code", ncclGetErrorString((ncclResult_t)-1));
};
