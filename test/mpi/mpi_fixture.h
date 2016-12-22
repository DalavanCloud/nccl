#pragma once
#include <map>
#include <mpi.h>
#include <vector>
template <typename T>
class mpi_test : public ::testing::Test {
    //
  public:
    static ncclUniqueId commId;
    static ncclComm_t comm;
    static cudaStream_t stream;
    static const int count1;
    static int countN;
    static T *buf_send_d, *buf_recv_d;
    static std::vector<T> buf_send_h, buf_recv_h, buf_recv_mpi;
    static const ncclDataType_t ncclDataType;
    static const MPI_Datatype mpiDataType;
    static const std::vector<ncclRedOp_t> RedOps;
    static const std::map<ncclRedOp_t, MPI_Op> MpiOps;
    static void SetUpTestCase();
    static void TearDownTestCase();
    virtual void SetUp();
    virtual void TearDown();
    //
  protected:
    void InitInput();
    void Verify(const int length) const;
    //
  private:
    static void AllocData();
    static void FreeData();
};
typedef ::testing::Types<char, int, float, double, long long,
                         unsigned long long>
    DTS;
TYPED_TEST_CASE(mpi_test, DTS);
