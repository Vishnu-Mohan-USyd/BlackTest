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

    auto *rgcPinsL = new queue<float**>;
    auto *rgcPinsR = new queue<float**>;;
    visualPass1(rgcPinsL, rgcPinsR, ref(rgcMutex), ref(rgcCond));
    // rgcPins->pop();
    cout << rgcPinsL->front()[6][8] << endl;

}
