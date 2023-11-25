#include <iostream>
#include "book.h"
__global__ void hello()
{
    printf("Hello CUDA\n");
}


int main() {
    hello<<<1,1>>>();
    cudaDeviceSynchronize();
    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));

    printf("cudaGetDeviceCount:%d",count);
    return 0;


}
