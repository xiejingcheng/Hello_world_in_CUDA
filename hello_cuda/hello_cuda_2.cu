#include <iostream>
#include "../book.h"
#include <cuda_runtime.h>
#define N (33 * 1024)

__global__ void add(int*a ,int *b, int *c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N){
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(){
    int a[N], b[N], c[N];
    int *dev_a ,*dev_b ,*dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_a ,N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b ,N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c ,N * sizeof(int)));

    HANDLE_ERROR(cudaMemset(dev_a, 0, N * sizeof(int)));
    HANDLE_ERROR(cudaMemset(dev_b, 0, N * sizeof(int)));
    HANDLE_ERROR(cudaMemset(dev_c, 0, N * sizeof(int)));

    for(int i=0 ;i<N;i++){
        a[i] = i;
        b[i] = i*i;
    }

    //注意这里的 dev_a a 方向不要弄反了
    HANDLE_ERROR(cudaMemcpy(dev_a,
                            a,
                            N * sizeof(int),
                            cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(dev_b,
                            b,
                            N * sizeof(int),
                            cudaMemcpyHostToDevice));
    
    add<<<128,128>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c,
                            dev_c,
                            N * sizeof(int),
                            cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i<N; i++){
        if((a[i]+b[i])!=c[i]){
            success = false;
        }
    }
    if(success)
    printf("success");

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}