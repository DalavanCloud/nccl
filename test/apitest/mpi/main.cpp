#include "TEST_ENV.h"
#include "mpi_fixture.h"
#include <iostream>
const int MAX_GPU_COUNT = 128;
bool isPerf = false;
int gpuList[MAX_GPU_COUNT] = {0};
void ParseArgs(int argc, char** argv) {
    for (int i = 0; i < argc; ++i) {
        if (0 == strcmp("-perf", argv[i])) {
            isPerf = true;
            break;
        } else if (0 == strcmp("-gpu", argv[i])) {
            // format: -gpu 0 1 2 1 0 //
            // the node list and ranks per node values are passed to mpirun.
            // so the gpu list is, the device id of each rank.
            ++i;
            for (int j = 0; (j < MAX_GPU_COUNT) && (i < argc); ++i, ++j) {
                if ('-' == argv[i][0])
                    break;
                gpuList[j] = atoi(argv[i]);
            }
        }
    }
}
int main(int argc, char** argv) {
    for (int i = 0; i < MAX_GPU_COUNT; ++i) {
        gpuList[i] = i;
    }
    ParseArgs(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new TEST_ENV); // it's deleted by gtest. don't delete it again.
    return RUN_ALL_TESTS();
}
