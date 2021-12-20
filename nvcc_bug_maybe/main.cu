#include <stdio.h>

template <class T>
class Vector
{
   public:
    __host__ __device__ Vector() : m_size(0)
    {
    }

    __host__ __device__ Vector(const Vector& rhs)
    {
        m_size = rhs.m_size;
    }

    __host__ __device__ Vector& operator=(const Vector& rhs)
    {
        m_size = rhs.m_size;
        return *this;
    }

    __host__ __device__ ~Vector()
    {
    }

    uint32_t m_size;
};

template <typename T>
__global__ void mainKernel(Vector<T> vec)
{
    // Change this lambda function to capture by reference ([&]) fixes the bug
    auto print_vector_size = [=]() {
        printf("\n vec size = %d\n", vec.m_size);
       int y = 0;
       int x = y == 0 ?: 1; // only works on GCC
    };

    print_vector_size();    
}

int main(int argc, char** argv)
{
    Vector<int> vec;
    vec.m_size = 10;

    mainKernel<<<1, 1>>>(vec);

    cudaDeviceSynchronize();
}
