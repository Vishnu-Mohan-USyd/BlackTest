#include <cstdio>
#include <iostream>
#include <libavcodec/avcodec.h>
#include "functs.h"
#include <vector>
#include <chrono>
#include <thread>
#include <omp.h>
#include "cuda_profiler_api.h"
#include <GLFW/glfw3.h>
#include <cmath>
#include <matplot/matplot.h>
#include "functs.h"
#include "rgc.cuh"
#include <mutex>
#include "../neurons/Neuron.h"

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

void RGCprocessing (float** rgcInputsL, float** rgcInputsR, RGC** rgcdets){
    
}

int rgcSpikers (queue<float**> *rihq_l, queue<float**> *rihq_r, vid2rgcParams *v2rp, queue<RGC**> *rgcdetails, mutex &rgcMut, condition_variable &rgcCond){
    // ---------------- Transferring data from vid2rgc to rgcSpikers ---------------
    int *xWidthsHost, *yMatchesHost, rgcArrayHeight, RGCcount;
    float **rgcInputsHost_l, **rgcInputsHost_r;
    RGC** RGCdets;
    {
        unique_lock<mutex> lock(rgcMut);
        rgcCond.wait(lock, [&]{ return !rihq_l->empty();});
        rgcArrayHeight = *v2rp->rgcArrH;
        RGCcount = *v2rp->RGCcnt;
        xWidthsHost = v2rp->xWid;
        yMatchesHost = v2rp->yMat;
        RGCdets = (RGC**)malloc(rgcArrayHeight * sizeof(RGC*));
        for(int i = 0; i < rgcArrayHeight; i+=1){
            RGCdets[i] = (RGC*)malloc(xWidthsHost[i] * sizeof(RGC));
            memcpy(RGCdets[i], rgcdetails->front()[i], (xWidthsHost[i] * sizeof(RGC)));
        }
    }

    while(true){
        {
            unique_lock<mutex> lock(rgcMut);
            rgcCond.wait(lock, [&]{ return !rihq_l->empty();});

            // Copying over the results from v
            rgcInputsHost_l = (float**)malloc(rgcArrayHeight * sizeof(float*));
            rgcInputsHost_r = (float**)malloc(rgcArrayHeight * sizeof(float*));
            for(int i = 0; i < rgcArrayHeight; i+=1){
                rgcInputsHost_l[i] = (float*)malloc(xWidthsHost[i] * sizeof(float));
                memcpy(rgcInputsHost_l[i], rihq_l->front()[i], (xWidthsHost[i] * sizeof(float)));
                rgcInputsHost_r[i] = (float*)malloc(xWidthsHost[i] * sizeof(float));
                memcpy(rgcInputsHost_r[i], rihq_r->front()[i], (xWidthsHost[i] * sizeof(float)));
            }
            rihq_l->pop();
            rihq_r->pop();
        }



    }

}


