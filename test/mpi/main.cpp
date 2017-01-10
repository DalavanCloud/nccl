#include "TEST_ENV.h"
#include "mpi_fixture.h"
bool isPerf = false;
void ParseArgs(int argc, char** argv) {
    for (int i = 0; i < argc; ++i) {
        if (0 == strcmp("-perf", argv[i])) {
            isPerf = true;
            break;
        }
    }
}
int main(int argc, char** argv) {
    ParseArgs(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(
        new TEST_ENV); // it's deleted by gtest. don't delete it again.
    return RUN_ALL_TESTS();
}
