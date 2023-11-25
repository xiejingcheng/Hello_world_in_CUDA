#include <iostream>
#include "../book.h"
#define N (33 * 1024)
#define imin(a,b) (a<b?a:b)

const int threadPerBlock = 128;

const int blockPerGrid = imin(32, (N + threadPerBlock-1)/threadPerBlock);

__global__ void dot(float*a ,float *b, float *c){
    __shared__ float cache[threadPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;
    float temp = 0.0;
    while (tid < N){
        temp += a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIdx] = temp;

    __syncthreads();

    int i = blockDim.x/2;
    while(i!=0){
        if(cacheIdx<i){
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
        i = i/2;
    }

    if(cacheIdx==0){
        c[blockIdx.x] = cache[0];
    }

}

int main(){
    float count = 0.0;
    float a[N], b[N], c[blockPerGrid];
    float *dev_a ,*dev_b ,*dev_c;
    
    HANDLE_ERROR(cudaMalloc((void**)&dev_a ,N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b ,N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c ,blockPerGrid * sizeof(float)));

    HANDLE_ERROR(cudaMemset(dev_a, 0, N * sizeof(float)));
    HANDLE_ERROR(cudaMemset(dev_b, 0, N * sizeof(float)));
    HANDLE_ERROR(cudaMemset(dev_c, 0, blockPerGrid * sizeof(float)));

    for(int i=0 ;i<N;i++){
        a[i] = i;
        b[i] = i*i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a,
        a,
        N * sizeof(float),
        cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(dev_b,
        b,
        N * sizeof(float),
        cudaMemcpyHostToDevice));

    dot<<<blockPerGrid, threadPerBlock>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c,
        dev_c,
        blockPerGrid * sizeof(float),
        cudaMemcpyDeviceToHost));
    
    for(int i =0 ;i < blockPerGrid; i++){
        count += c[i];
    }

    printf("%f",count);


    return 0;
}