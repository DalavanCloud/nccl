#pragma once
class mpi_test : public ::testing::Test {
    //
  public:
    //
  protected:
    static ncclUniqueId commId;
    static ncclComm_t comm;
    static cudaStream_t stream;
    static const int count = 128;
    static int *buf_send, *buf_recv, *buf_host;
    static void SetUpTestCase();
    static void TearDownTestCase();
};
