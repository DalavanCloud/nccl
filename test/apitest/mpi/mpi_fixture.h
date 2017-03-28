#pragma once
#include <chrono>
#include <map>
#include <mpi.h>
#include <vector>
extern bool isPerf;
template <typename T>
class mpi_test : public ::testing::Test {
    //
  public:
    static ncclUniqueId commId;
    static ncclComm_t comm;
    static cudaStream_t stream;
    static long long int count1;
    static long long int countN;
    static T *buf_send_d, *buf_recv_d;
    static std::vector<T> buf_send_h, buf_recv_h, buf_recv_mpi;
    static const ncclDataType_t ncclDataType;
    static const MPI_Datatype mpiDataType;
    static const std::vector<ncclRedOp_t> RedOps;
    static const std::map<ncclRedOp_t, MPI_Op> MpiOps;
    std::chrono::system_clock::time_point t_start;
    std::chrono::system_clock::time_point t_end;
    double t_duration = 0;
    int loop_count =
        1; // if perf test, loop many times, in order to get average value.
    int loop_factor = 1; // some test have several operations.
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
//
#define PERF_BEGIN()                                                           \
    for (int i = 0; i <= this->loop_count; ++i) {                              \
        if (i == 1) {                                                          \
            MPI_Barrier(MPI_COMM_WORLD);                                       \
            this->t_start = std::chrono::high_resolution_clock::now();         \
        }
//
#define PERF_END()                                                             \
    }                                                                          \
    this->t_end = std::chrono::high_resolution_clock::now();                   \
    this->t_duration =                                                         \
        (double)(std::chrono::duration_cast<std::chrono::milliseconds>(        \
                     this->t_end - this->t_start)                              \
                     .count()) /                                               \
        (double)this->loop_count / (double)this->loop_factor;
// EOF
