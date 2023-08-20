#include <cstdio>
#include <iostream>
#include <libavcodec/avcodec.h>
#include "functs.h"
#include <vector>
#include <chrono>
#include "../vidStuff/vidReader.h"
#include <thread>
#include <omp.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <matplot/matplot.h>
#include "rgc.cuh"
#include "mainseq.h"

using namespace matplot;
using namespace std;
RGCinitVals initSizes;

void visualEngine(){

    float **rgcInputPin;
    initSizes = retrgcinits();

    rgcInputPin = (float**)malloc(initSizes.rgcArrayH * sizeof(float*));

    for(int i = 0; i < initSizes.rgcArrayH; i+=1){
        rgcInputPin[i] = (float*)malloc(initSizes.xWidths[i] * sizeof(float));
    }
    visualPass1(rgcInputPin, initSizes.xWidths, initSizes.rgcArrayH, initSizes.yMatches, initSizes.RGCcount);

}
