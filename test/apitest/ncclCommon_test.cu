#include "ncclCommon_test.cuh"
// these are Template specialization
#define GEN_DATATYPE(X, Y)                                                     \
    template <>                                                                \
    ncclDataType_t ncclCommon_test<X>::DataType() {                            \
        return (Y);                                                            \
    };
GEN_DATATYPE(char, ncclChar);
GEN_DATATYPE(half, ncclHalf);
GEN_DATATYPE(int, ncclInt);
GEN_DATATYPE(float, ncclFloat);
GEN_DATATYPE(double, ncclDouble);
GEN_DATATYPE(long long, ncclInt64);
GEN_DATATYPE(unsigned long long, ncclUint64);
#undef GEN_DATATYPE
