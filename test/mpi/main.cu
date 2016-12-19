#include "TEST_ENV.h"
#include "mpi_fixture.h"
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(
        new TEST_ENV); // it's deleted by gtest. don't delete it again.
    return RUN_ALL_TESTS();
}
