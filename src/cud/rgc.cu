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

vid_rgc_params vidRgcParams;
//
// Created by kasm-user on 7/26/23.
//


__global__
void RGCprocessing (float** rgcInputsL, float** rgcInputsR, RGC** rgcdets, vid_rgc_params vidParams ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Gets the current coords in the RGC array
    int tempCumWidth = 0, posX = 0, posY = 0;
    for(int j = 0; j < vidParams.rgcArrayHeight; j+=1){
        tempCumWidth += vidParams.xWidths[j];
        if(i < tempCumWidth){
            posY = j;
            posX = i - (tempCumWidth - vidParams.xWidths[j]);
            break;
        }
    }
    if(rgcdets[posX][posY].type == MIDGET){


    } else if(rgcdets[posX][posY].type == PARASOL){

    } else if(rgcdets[posX][posY].type == SBC){

    }

}

int rgcSpikers (queue<float**> *rihq_l, queue<float**> *rihq_r, vid2rgcParams *v2rp, queue<RGC**> *rgcdetails, mutex &rgcMut, condition_variable &rgcCond){
    // ---------------- Transferring data from vid2rgc to rgcSpikers ---------------
    int *xWidthsHost, *yMatchesHost, rgcArrayHeight, RGCcount, *xWid_d, *yMat_d;
    float **rgcInputsHost_l, **rgcInputsHost_r;
    RGC** RGCdets;

    // ---------------- Initial prep - during first load only -----------------------------
    {
        unique_lock<mutex> lock(rgcMut);
        rgcCond.wait(lock, [&]{ return !rihq_l->empty();});
        vidRgcParams.rgcArrayHeight = *v2rp->rgcArrH;
        vidRgcParams.RGCcount = *v2rp->RGCcnt;
        vidRgcParams.fps = 30;
        xWidthsHost = v2rp->xWid;
        yMatchesHost = v2rp->yMat;
        RGCdets = (RGC**)malloc(rgcArrayHeight * sizeof(RGC*));
        for(int i = 0; i < rgcArrayHeight; i+=1){
            RGCdets[i] = (RGC*)malloc(xWidthsHost[i] * sizeof(RGC));
            memcpy(RGCdets[i], rgcdetails->front()[i], (xWidthsHost[i] * sizeof(RGC)));
        }
        cudaMalloc(&xWid_d, vidRgcParams.rgcArrayHeight * sizeof(int));
        cudaMalloc(&yMat_d, *v2rp->perspH * sizeof(int));
        cudaMemcpy(xWid_d, xWidthsHost, vidRgcParams.rgcArrayHeight * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(yMat_d, yMatchesHost, *v2rp->perspH * sizeof(int), cudaMemcpyHostToDevice);
        vidRgcParams.xWidths = xWid_d;
        vidRgcParams.yMatches = yMat_d;
    }
    // ---------------------------------- X -----------------------------------

    // ----------------------- Program Loop ------------------------------------

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

    // -------------------------- End of program Loop --------------------------------

}


