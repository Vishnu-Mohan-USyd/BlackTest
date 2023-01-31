#include <cstdio>
#include <iostream>
#include <libavcodec/avcodec.h>
#include "functs.h"
#include <vector>

using namespace std;
__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
}

__global__
void arrInit(int n, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
}

int mrain(vector<vector<float>> &gcuArr)
{
    int N = 1<<26;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int i = 0; i < deviceCount; i+=1){
        vector<float> temp(1<<26, 5);
        gcuArr.push_back(temp);
    }
    gcuArr[1][3] = 4;
    float *d_x, *d_y;

    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));


    cudaMemcpy(d_x, gcuArr[0].data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, gcuArr[1].data(), N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    for (int i = 0; i < 4; i+=1){
        saxpy<<<(N+1023)/1024, 1024>>>(N, 2.0f, d_x, d_y);
    }


    cudaMemcpy(gcuArr[1].data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceProp a{};

    cudaGetDeviceProperties(&a, 0);
    std::cout << "Number of devices :         " << deviceCount << endl;
    std::cout << "Device type :               " << a.managedMemory << endl;
    std::cout << "Test var 3 :                " << gcuArr[1][4] << endl;

    cudaSetDevice(0);

    float maxError = 0.0f;
//    for (int i = 0; i < N; i++)
//        maxError = max(maxError, abs(y[i]-4.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
}