#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <nvfunctional>
#include "gtest/gtest.h"

__device__ void call_print(nvstd::function<void(int&, bool&)> print_fun)
{
    int  val = 10;
    bool check = true;
    print_fun(val, check);
}

__global__ void mainKernel()
{

    auto print_me = [=](int& val, bool& check) {
        if (check) {
            printf("\n val = %d\n", val);
        }
    };

    call_print(print_me);
}

TEST(Test, simple)
{
    mainKernel<<<1, 1>>>();
    auto err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
