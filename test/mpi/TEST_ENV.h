#pragma once
class TEST_ENV : public ::testing::Environment {
  public:
    static int gpu_count;
    static int mpi_size;
    static int mpi_rank;
    static const int mpi_root = 0;
    TEST_ENV(){};
    virtual ~TEST_ENV(){};
    void SetGPU();
    // Override this to define how to set up the environment.
    virtual void SetUp();
    // Override this to define how to tear down the environment.
    virtual void TearDown();
};
