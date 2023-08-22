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
#include <mutex>

using namespace matplot;
using namespace std;

RGCinitVals  rets;
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

int rgcSpikers (RGCinitVals rgcInputs, PARAMS rgcparams){

}


