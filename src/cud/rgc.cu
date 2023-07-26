#include <cstdio>
#include <iostream>
#include <libavcodec/avcodec.h>
#include "functs.h"
#include <vector>
#include <chrono>
#include "../vidStuff/vidReader.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thread>
#include <omp.h>
#include "cuda_profiler_api.h"
#include <GLFW/glfw3.h>
#include <cmath>
#include <matplot/matplot.h>
#include "functs.h"
#include "rgc.cuh"

using namespace matplot;
using namespace std;

RGCdev  rets;
//
// Created by kasm-user on 7/26/23.
//
__global__
void saxpy(double *x, float ** da)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // da[3][4] = 87;
    x[i] = da[76][88];


}

int rgcSpikers (){
    double *x, *xTemp;
    xTemp = (double*)malloc(10000 * sizeof(double));
    cudaSetDevice(0);
    cudaStream_t str1;
    cudaStreamCreate (&str1);
    cudaDeviceSynchronize();
    cudaMalloc(&x, 10000 * sizeof(double));
    cudaMallocHost((void**)&xTemp, 10000 * sizeof(double));
    float **parvoLeftDev, **parvoRightDev, **parvoLeftHost, **parvoRightHost, **parvoPin,
            **magnoLeftDev, **magnoRightDev, **magnoLeftHost, **magnoRightHost;
    rets = visualPass1();
    parvoPin = rets.parvo;
    saxpy<<<(10000 + 1023)/1024, 1024, 0, str1>>>(x, parvoPin);
    cudaDeviceSynchronize();
    cudaMemcpy(xTemp, x, 10000 * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << xTemp[8]  << std::endl;
}


