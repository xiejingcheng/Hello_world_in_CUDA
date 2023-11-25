#include <stdio.h>
#include <cuda.h>

__global__ void hello()
{
    printf("Hello CUDA\n");
}

int main() {
    hello<<<1,1000 >>>();
    cudaDeviceSynchronize();
    return 0;
}
