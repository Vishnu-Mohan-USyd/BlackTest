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
#include <queue>
#include "mainseq.h"
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace matplot;
using namespace std;
RGCinitVals initSizes;

mutex rgcMutex;
condition_variable rgcCond;

void visualEngine(){

    int frn = 0;
    int *frameNumber = (int*)malloc(sizeof(int));
    *frameNumber = 0;
    auto *rgcPinsL = new queue<float**>;
    auto *rgcPinsR = new queue<float**>;
    auto *rgcDetsPin = new queue<RGC**>;
    vid2rgcParams *v2rp = (vid2rgcParams*) malloc(4 * sizeof(int*));
    v2rp->RGCcnt = (int*)malloc(sizeof(int));
    v2rp->rgcArrH = (int*)malloc(sizeof(int));
    v2rp->perspH = (int*)malloc(sizeof(int));
    thread vid2rgcThread(vid2rgc, frameNumber, rgcPinsL, rgcPinsR, rgcDetsPin, v2rp, ref(rgcMutex), ref(rgcCond));
    // rgcPins->pop();
    thread rgc2lgnThread(rgcSpikers, rgcPinsL, rgcPinsR, v2rp, rgcDetsPin, ref(rgcMutex), ref(rgcCond));
    vid2rgcThread.join();
    rgc2lgnThread.join();
//    while(true){
//        unique_lock<mutex> lock(rgcMutex);
//        rgcCond.wait(lock, [&]{ return !rgcPinsL->empty();});
//        cout << *frameNumber << " xWid[4] : " << *v2rp->RGCcnt << endl;
//    }


}
