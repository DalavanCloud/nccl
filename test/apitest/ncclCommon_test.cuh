#pragma once
template <typename OP, typename DT>
void freePP(OP op, DT**& ptr, const int len) {
    if (ptr != NULL) {
        for (int i = 0; i < len; ++i) {
            EXPECT_EQ(cudaSuccess, cudaSetDevice(i));
            EXPECT_NO_FATAL_FAILURE(op(ptr[i]));
            ptr[i] = NULL;
        };
        free(ptr);
        ptr = NULL;
    };
};
template <typename DT>
class ncclCommon_test : public ::testing::Test {
  public:
    static int N;
    static int nVis;
    static ncclComm_t* comms;
    static DT **sendbuffs, **recvbuffs, //
        **sendbuffs_host, **recvbuffs_host, //
        **sendbuffs_pinned, **recvbuffs_pinned;
    static cudaStream_t* streams;
    static ncclDataType_t DataType();
    static void SetUpTestCase();
    static void TearDownTestCase();
    static const std::vector<ncclRedOp_t> RedOps;
    //
  protected:
    int root = -1;
    void SetUp(){};
    void TearDown() {
        for (int i = 0; i < this->nVis; ++i) {
            EXPECT_EQ(cudaSuccess, cudaSetDevice(i));
            EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->streams[i]));
        }
    };
};
template <typename DT>
const std::vector<ncclRedOp_t> ncclCommon_test<DT>::RedOps = {ncclSum, ncclProd,
                                                              ncclMax, ncclMin};
template <typename DT>
int ncclCommon_test<DT>::N = 1000;
template <typename DT>
int ncclCommon_test<DT>::nVis = -1;
template <typename DT>
ncclComm_t* ncclCommon_test<DT>::comms = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::sendbuffs = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::recvbuffs = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::sendbuffs_host = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::recvbuffs_host = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::sendbuffs_pinned = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::recvbuffs_pinned = NULL;
template <typename DT>
cudaStream_t* ncclCommon_test<DT>::streams = NULL;
template <typename DT>
void ncclCommon_test<DT>::SetUpTestCase() {
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
    if (nVis < 2) {
        FAIL() << "waived: not enough gpu";
    };
    streams = (cudaStream_t*)calloc(nVis, sizeof(cudaStream_t));
    sendbuffs = (DT**)calloc(nVis, sizeof(DT**));
    recvbuffs = (DT**)calloc(nVis, sizeof(DT**));
    sendbuffs_host = (DT**)calloc(nVis, sizeof(DT**));
    recvbuffs_host = (DT**)calloc(nVis, sizeof(DT**));
    sendbuffs_pinned = (DT**)calloc(nVis, sizeof(DT**));
    recvbuffs_pinned = (DT**)calloc(nVis, sizeof(DT**));
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&sendbuffs[i], N * sizeof(DT)));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&recvbuffs[i], N * sizeof(DT)));
        ASSERT_EQ(cudaSuccess, cudaMemset(sendbuffs[i], N * sizeof(DT), 0));
        ASSERT_EQ(cudaSuccess, cudaMemset(recvbuffs[i], N * sizeof(DT), 0));
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i])) << i;
        sendbuffs_host[i] = (DT*)calloc(N, sizeof(DT));
        recvbuffs_host[i] = (DT*)calloc(N, sizeof(DT));
        sendbuffs_pinned[i] = (DT*)calloc(N, sizeof(DT));
        ASSERT_EQ(cudaSuccess,
                  cudaHostRegister(sendbuffs_pinned[i], N * sizeof(DT),
                                   cudaHostRegisterDefault));
        recvbuffs_pinned[i] = (DT*)calloc(N, sizeof(DT));
        ASSERT_EQ(cudaSuccess,
                  cudaHostRegister(recvbuffs_pinned[i], N * sizeof(DT),
                                   cudaHostRegisterDefault));
    }
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
};
template <typename DT>
void ncclCommon_test<DT>::TearDownTestCase() {
    EXPECT_NO_FATAL_FAILURE(
        freePP<>([](ncclComm_t ptr) { ncclCommDestroy(ptr); }, comms, nVis));
    auto freecuda = [](DT* ptr) { cudaFree(ptr); };
    EXPECT_NO_FATAL_FAILURE(freePP<>(freecuda, sendbuffs, nVis));
    EXPECT_NO_FATAL_FAILURE(freePP<>(freecuda, recvbuffs, nVis));
    auto freehost = [](DT* ptr) { free(ptr); };
    EXPECT_NO_FATAL_FAILURE(freePP<>(freehost, sendbuffs_host, nVis));
    EXPECT_NO_FATAL_FAILURE(freePP<>(freehost, recvbuffs_host, nVis));
    auto freePinned = [](DT* ptr) {
        EXPECT_EQ(cudaSuccess, cudaHostUnregister(ptr));
        free(ptr);
    };
    EXPECT_NO_FATAL_FAILURE(freePP<>(freePinned, sendbuffs_pinned, nVis));
    EXPECT_NO_FATAL_FAILURE(freePP<>(freePinned, recvbuffs_pinned, nVis));
    auto freeStream = [](cudaStream_t st) { cudaStreamDestroy(st); };
    EXPECT_NO_FATAL_FAILURE(freePP<>(freeStream, streams, nVis));
};
typedef ::testing::Types<char, int, float, double, long long,
                         unsigned long long>
    testDataTypes;
/// TODO: half type causes compilation error.
// TYPED_TEST_CASE(ncclCommon_test, testDataTypes);
// EOF
