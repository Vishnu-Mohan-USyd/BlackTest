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
#include "rgc.cuh"
#include <thread>
#include <random>

using namespace matplot;
using namespace std;

PARAMS params;
FRUSTUM frustum;
RGCPARAMS rgcparams;

__global__
void saxpy(RGC** RGCdet_r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Gets the current coords in the RGC array


}

void CalcFrustum(void)
{
    double dh,dv;

    XYZ vp = {0,0,0};     // At the origin
    XYZ vd = {0,1,0};     // View Direction. Looking down the +y axis
    XYZ vu = {0,0,1};     // Up vector. z axis
    XYZ vr = {1,0,0};     // Right vector, x axis. CrossProduct(vd,vu);

    dh = tan(params.perspfov * (M_PI / 180) / 2);                 // half frustum width
    dv = params.perspHeight * dh / params.perspWidth;     // half frustum height

    // Corners of view frustum
    frustum.p1 = VectorSum(1.0,vp,1.0,vd,-dh,vr, dv,vu);
    frustum.p2 = VectorSum(1.0,vp,1.0,vd,-dh,vr,-dv,vu);
    frustum.p3 = VectorSum(1.0,vp,1.0,vd, dh,vr,-dv,vu);
    frustum.p4 = VectorSum(1.0,vp,1.0,vd, dh,vr, dv,vu);

    // Apply horizontal and vertical offset
    //
    // Offset = 0                 Offset = 1
    //            +                   +
    //          / |                 / |
    //        /   |                /  |
    //      /     |               /   |
    //    --------+---> y        /    |
    //      \     |             /     |
    //        \   |            /      |
    //          \ |           --------+---> y
    //            +
    //
    frustum.p1.x += dv * params.hoffset * vr.x;
    frustum.p1.z += dv * params.voffset * vu.z;
    frustum.p2.x += dv * params.hoffset * vr.x;
    frustum.p2.z += dv * params.voffset * vu.z;
    frustum.p3.x += dv * params.hoffset * vr.x;
    frustum.p3.z += dv * params.voffset * vu.z;
    frustum.p4.x += dv * params.hoffset * vr.x;
    frustum.p4.z += dv * params.voffset * vu.z;
}

__device__
XYZ CameraRay(double x,double y, PARAMS deviceParams, FRUSTUM frust)
{
    int k;
    double u,v;
    XYZ p,q;

    u = (double)x / deviceParams.perspWidth;
    v = (double)(deviceParams.perspHeight - (double)y) / (double)deviceParams.perspHeight;

    p.x = frust.p1.x + u * (frust.p4.x - frust.p1.x);
    p.y = frust.p1.y;
    p.z = frust.p1.z + v * (frust.p2.z - frust.p1.z);

    // Apply rotations
    for (k=0;k<deviceParams.ntransform;k++) {
        switch(deviceParams.transform[k].axis) {
            case XTILT:
                q.x =  p.x;
                q.y =  p.y * deviceParams.transform[k].cvalue + p.z * deviceParams.transform[k].svalue;
                q.z = -p.y * deviceParams.transform[k].svalue + p.z * deviceParams.transform[k].cvalue;
                break;
            case YROLL:
                q.x =  p.x * deviceParams.transform[k].cvalue + p.z * deviceParams.transform[k].svalue;
                q.y =  p.y;
                q.z = -p.x * deviceParams.transform[k].svalue + p.z * deviceParams.transform[k].cvalue;
                break;
            case ZPAN:
                q.x =  p.x * deviceParams.transform[k].cvalue + p.y * deviceParams.transform[k].svalue;
                q.y = -p.x * deviceParams.transform[k].svalue + p.y * deviceParams.transform[k].cvalue;
                q.z =  p.z;
                break;
        }
        p = q;
    }

    return(p);
}

XYZ VectorSum(double d1,XYZ p1,double d2,XYZ p2,double d3,XYZ p3,double d4,XYZ p4)
{
    XYZ sum;

    sum.x = d1 * p1.x + d2 * p2.x + d3 * p3.x + d4 * p4.x;
    sum.y = d1 * p1.y + d2 * p2.y + d3 * p3.y + d4 * p4.y;
    sum.z = d1 * p1.z + d2 * p2.z + d3 * p3.z + d4 * p4.z;

    return(sum);
}

__global__
void createPerspTest(uint8_t  *perspTest, uint8_t  *perspLeft, PARAMS deviceParams){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int x = i % deviceParams.perspWidth;
    int y = i / deviceParams.perspWidth;

//    if(true){
//        perspTest[(i * 4)] = perspLeft[(i * 4)];
//        perspTest[(i * 4) + 1] = perspLeft[(i * 4) + 1];
//        perspTest[(i * 4) + 2] = perspLeft[(i * 4) + 2];
//        perspTest[(i * 4) + 3] = perspLeft[(i * 4) + 3];
//    } else {
//        perspTest[(i * 4)] = 0;
//        perspTest[(i * 4) + 1] = 0;
//        perspTest[(i * 4) + 2] = 0;
//        perspTest[(i * 4) + 3] = 0;
//    }

    perspTest[(i * 4)] = 0;
    perspTest[(i * 4) + 1] = 0;
    perspTest[(i * 4) + 2] = 0;
    perspTest[(i * 4) + 3] = 0;

//        if((x < 900)){
//            perspTest[(i * 4)] = perspLeft[(i * 4)];
//            perspTest[(i * 4) + 1] = perspLeft[(i * 4) + 1];
//            perspTest[(i * 4) + 2] = perspLeft[(i * 4) + 2];
//            perspTest[(i * 4) + 3] = perspLeft[(i * 4) + 3];
//    } else {
////        perspLeft[(i * 4)] = 0;
////        perspLeft[(i * 4) + 1] = 0;
////        perspLeft[(i * 4) + 2] = 0;
////        perspLeft[(i * 4) + 3] = 0;
//    }

//    if(((x / 50) % 2 == 0) && ((y / 50) % 2 == 0)){
//        perspLeft[(i * 4)] = 0;
//        perspLeft[(i * 4) + 1] = 255;
//        perspLeft[(i * 4) + 2] = 0;
//        perspLeft[(i * 4) + 3] = 255;
//    } else {
//        perspLeft[(i * 4)] = 0;
//        perspLeft[(i * 4) + 1] = 0;
//        perspLeft[(i * 4) + 2] = 0;
//        perspLeft[(i * 4) + 3] = 0;
//    }
}

__global__
void world2Persp(::uint8_t  *worldLeft, uint8_t  *perspLeft, ::uint8_t  *worldRight, uint8_t  *perspRight, PARAMS deviceParams, FRUSTUM deviceFrust)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int x = i % deviceParams.perspWidth;
    int y = i / deviceParams.perspWidth;




    RGB rgbLeft = {0, 0, 0, 0}, rgbRight = {0, 0, 0, 0};
    XYZ par;
    double longitude, latitude, xAlias, yAlias, x_sphere, y_sphere;
    int top_left_x, top_left_y, spIndex;


    for (int ai=0;ai<deviceParams.antialias;ai++) {
        xAlias = x + ai / (double)deviceParams.antialias;
        for (int aj=0;aj<deviceParams.antialias;aj++) {
            yAlias = y + aj / (double) deviceParams.antialias;
            par = CameraRay((double) xAlias, (double) yAlias, deviceParams, deviceFrust);
            longitude = atan2(par.x, par.y);                     // -pi ... pi
            latitude = atan2(par.z, sqrt(par.x * par.x + par.y * par.y));    // -pi/2 ... pi/2
            x_sphere = (longitude - deviceParams.longmin) * deviceParams.worldWidth /
                       (deviceParams.longmax - deviceParams.longmin);
            y_sphere = (latitude - deviceParams.latmin) * deviceParams.worldHeight /
                       (deviceParams.latmax - deviceParams.latmin);
            if (x_sphere < 0 || y_sphere < 0)
                continue;
            if (y_sphere >= deviceParams.worldHeight)
                continue;

            if (x_sphere >= deviceParams.worldWidth) {
                if (deviceParams.longmin == -M_PI && deviceParams.longmax == M_PI)
                    x_sphere -= deviceParams.worldWidth;
                else
                    continue;
            }

            top_left_x = (int) x_sphere;
            top_left_y = (int) y_sphere;

            spIndex = (top_left_y * deviceParams.worldWidth * 4) + (top_left_x * 4);
            rgbLeft.r += worldLeft[spIndex];
            rgbLeft.g += worldLeft[spIndex + 1];
            rgbLeft.b += worldLeft[spIndex + 2];
            rgbLeft.a += worldLeft[spIndex + 3];
            rgbRight.r += worldRight[spIndex];
            rgbRight.g += worldRight[spIndex + 1];
            rgbRight.b += worldRight[spIndex + 2];
            rgbRight.a += worldRight[spIndex + 3];

        }
    }
    perspLeft[(i * 4)] = rgbLeft.r / deviceParams.antialias2;
    perspLeft[(i * 4) + 1] = rgbLeft.g / deviceParams.antialias2;
    perspLeft[(i * 4) + 2] = rgbLeft.b / deviceParams.antialias2;
    perspLeft[(i * 4) + 3] = rgbLeft.a / deviceParams.antialias2;
    perspRight[(i * 4)] = rgbRight.r / deviceParams.antialias2;
    perspRight[(i * 4) + 1] = rgbRight.g / deviceParams.antialias2;
    perspRight[(i * 4) + 2] = rgbRight.b / deviceParams.antialias2;
    perspRight[(i * 4) + 3] = rgbRight.a / deviceParams.antialias2;
    // testboy.testr = 56;
    // perspLeft[i] = deviceParams->testr;

}
__global__
void formRGCcurrents_l(RGCPARAMS rgcparams, uint8_t *perspTest, uint8_t  *perspLeft, float** leftInputs, int* xWidths, int* yMatch, RGC** RGCdet_l, int frameNum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Gets the current coords in the RGC array
    int tempCumWidth = 0, posX = 0, posY = 0, internalIndex = 0, tempParasolIndex = 0, parasolWeight = 3;
    for(int j = 0; j < rgcparams.rgcArrLen; j+=1){
        tempCumWidth += xWidths[j];
        if(i < tempCumWidth){
            posY = j;
            posX = i - (tempCumWidth - xWidths[j]);
            break;
        }
    }

    float surrSumL = 0, cenSumL = 0, surrIdeal = 0, cenIdeal = 0, surrValL = 0, cenValL = 0,
            xComp, yComp;
    int midX, midY, surrSide = (RGCdet_l[posY][posX].cenRfSide + (2 * RGCdet_l[posY][posX].surRfWidth)), currIndex, cenIndex, testr = 0;
    if(surrSide % 2 == 0) {
        midX = surrSide / 2;
        midY = midX;
    }
    if(surrSide % 2 == 1) {
        midX = (surrSide / 2) + 1;
        midY = midX;
    }

    cenIndex = (RGCdet_l[posY][posX].perspY * 4 * rgcparams.perspWidth) + (RGCdet_l[posY][posX].perspX * 4);


    for (int y = 1; y < (surrSide) + 1; y+=1){
        // Y boundary condition
        if ((((RGCdet_l[posY][posX].perspY < midY) && (y < midY) && ((midY - y) > RGCdet_l[posY][posX].perspY)) ||
             ((((rgcparams.perspHeight - 1) - (RGCdet_l[posY][posX].perspY)) < midY) && (y > midY) &&
              ((y - midY) > ((rgcparams.perspHeight - 1) - (RGCdet_l[posY][posX].perspY)))))) {
            continue;
        }
        for(int x = 1; x < (surrSide) + 1; x += 1){
            // X boundary condition
            if ((((RGCdet_l[posY][posX].perspX < midX) && (x < midX) && ((midX - x) > RGCdet_l[posY][posX].perspX)) ||
                 ((((rgcparams.perspWidth - 1) - (RGCdet_l[posY][posX].perspX)) < midX) && (x > midX) &&
                  ((x - midX) > ((rgcparams.perspWidth - 1) - (RGCdet_l[posY][posX].perspX)))))) {
                continue;
            }
            internalIndex = ((y - 1) * surrSide) + (x - 1);
            currIndex = cenIndex + ((y - midY) * 4 * rgcparams.perspWidth) + ((x - midX) * 4);
            xComp = 0; yComp = 0;

            // --------------------------- Surround Region -------------------------------
            if((x <= (RGCdet_l[posY][posX].surRfWidth)) || (x > (surrSide - RGCdet_l[posY][posX].surRfWidth)) ||
               (y <= (RGCdet_l[posY][posX].surRfWidth)) || (y > (surrSide - RGCdet_l[posY][posX].surRfWidth))){
                xComp = (float)x;
                yComp = (float)y;
                if(x > midX){
                    xComp = (float)((surrSide + 1) - x);
                }
                if(y > midY){
                    yComp = (float)((surrSide + 1) - y);
                }
                if (RGCdet_l[posY][posX].detType == LUM){
                    testr+=1;
                    surrSumL += (float)(xComp + yComp) * (float)perspLeft[currIndex + 3];
                    surrIdeal += (float)(xComp + yComp) * 255;
                }  // Parasol Cells
                else if (RGCdet_l[posY][posX].detType == MOTION) {
                    if(frameNum == 0){
                        RGCdet_l[posY][posX].previousValues[internalIndex] = (float)perspLeft[currIndex + 3];
                        RGCdet_l[posY][posX].parasolWeights_curr[internalIndex] = 0;
                        surrSumL = 0; surrIdeal = 255;
                    }else {
                        // Checks to see if ON / OFF condition is met

                        if(perspLeft[currIndex + 3] > RGCdet_l[posY][posX].previousValues[internalIndex]) {
                            surrSumL += ((float)perspLeft[currIndex + 3] - (float)RGCdet_l[posY][posX].previousValues[internalIndex]) +
                                       (((float)perspLeft[currIndex + 3] - (float)RGCdet_l[posY][posX].previousValues[internalIndex]) *
                                        ((float)RGCdet_l[posY][posX].parasolWeights_curr[internalIndex] / 10));
                            surrIdeal += 255 + (255 * ((float)RGCdet_l[posY][posX].parasolWeights_curr[internalIndex] / 10));

                            // Spreading weights to nearby units
                            for(int w = 0; w < 9; w+=1){
                                // Boundary check
                                tempParasolIndex = (((y - 1) + ((w/3) - 1)) * surrSide) + ((x - 1) + ((w % 3) - 1));
                                if((((y) + ((int)(w/3) - 1)) == 0 || ((x) + ((int)(w % 3) - 1)) == 0 || ((y) + ((int)(w/3) - 1)) == surrSide + 1 || ((x) + ((int)(w % 3) - 1)) == surrSide + 1) ||
                                   (float)RGCdet_l[posY][posX].previousValues[tempParasolIndex] >= (float)perspLeft[currIndex + 3] ||
                                   RGCdet_l[posY][posX].perspX < 4 || RGCdet_l[posY][posX].perspY < 4 || RGCdet_l[posY][posX].perspX > rgcparams.perspWidth - 4 || RGCdet_l[posY][posX].perspY > rgcparams.perspHeight - 4){
                                    continue;
                                } else {
                                    // Check to make sure only greater shifts in weights are done
                                    if(RGCdet_l[posY][posX].parasolWeights_curr[tempParasolIndex] < (int)((((float)perspLeft[currIndex + 3] -
                                                                                                            (float)RGCdet_l[posY][posX].previousValues[tempParasolIndex]) / 255) * parasolWeight)){
                                        RGCdet_l[posY][posX].parasolWeights_curr[internalIndex] =
                                                (int)((((float)perspLeft[currIndex + 3] - (float)RGCdet_l[posY][posX].previousValues[internalIndex]) / 255) * parasolWeight);
                                    }
                                }
                            }
                        }
                    }
                }
                else if (RGCdet_l[posY][posX].detType == COLOR){
                    if(RGCdet_l[posY][posX].colID == R_rgc){
                        // Surround is -M                     // M
                        surrSumL += ((xComp + yComp) * (float)perspLeft[currIndex + 1]);
                        surrIdeal += (xComp + yComp) * 255;
                    } else if (RGCdet_l[posY][posX].colID == G_rgc){
                        // Surround is -(S + L)                                                 // S                              // L
                        surrSumL += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspLeft[currIndex + 2] : (float)perspLeft[currIndex]));
                        surrIdeal += (xComp + yComp) * 255;
                    } else if(RGCdet_l[posY][posX].colID == B_rgc){
                        // Surround is -L                     // L
                        surrSumL += ((xComp + yComp) * (float)perspLeft[currIndex]);
                        surrIdeal += (xComp + yComp) * 255;
                    } else if (RGCdet_l[posY][posX].colID == Y_rgc){
                        // Surround is -(S + M)                                                 // S                              // M
                        surrSumL += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspLeft[currIndex + 2] : (float)perspLeft[currIndex + 1]));
                        surrIdeal += (xComp + yComp) * 255;
                    }
                }
            }
                // ----------------------------- X -------------------------------
                // --------------------------- Center Region -------------------------------
            else {
                xComp = (float)x - RGCdet_l[posY][posX].surRfWidth;
                yComp = (float)y - RGCdet_l[posY][posX].surRfWidth;
                if(x > midX){
                    xComp = (float)((RGCdet_l[posY][posX].cenRfSide + RGCdet_l[posY][posX].surRfWidth + 1) - x);
                }
                if(y > midY){
                    yComp = (float)((RGCdet_l[posY][posX].cenRfSide + RGCdet_l[posY][posX].surRfWidth + 1) - y);
                }
                // Midget - Lum
                if (RGCdet_l[posY][posX].detType == LUM){
                    cenSumL += (float)(xComp + yComp) * (float)perspLeft[currIndex + 3];
                    cenIdeal += (float)(xComp + yComp) * 255;
                }
                    // Parasol Cells
                else if (RGCdet_l[posY][posX].detType == MOTION) {
                    if(frameNum == 0){
                        RGCdet_l[posY][posX].previousValues[internalIndex] = (float)perspLeft[currIndex + 3];
                        RGCdet_l[posY][posX].parasolWeights_curr[internalIndex] = 0;
                        cenSumL = 0; cenIdeal = 255;
                    }else {
                        // Checks to see if ON / OFF condition is met

                        if(perspLeft[currIndex + 3] > RGCdet_l[posY][posX].previousValues[internalIndex]) {
                            cenSumL += ((float)perspLeft[currIndex + 3] - (float)RGCdet_l[posY][posX].previousValues[internalIndex]) +
                                       (((float)perspLeft[currIndex + 3] - (float)RGCdet_l[posY][posX].previousValues[internalIndex]) *
                                        ((float)RGCdet_l[posY][posX].parasolWeights_curr[internalIndex] / 10));
                            cenIdeal += 255 + (255 * ((float)RGCdet_l[posY][posX].parasolWeights_curr[internalIndex] / 10));

                            // Spreading weights to nearby units
                            for(int w = 0; w < 9; w+=1){
                                // Boundary check
                                tempParasolIndex = (((y - 1) + ((w/3) - 1)) * surrSide) + ((x - 1) + ((w % 3) - 1));
                                if((((y) + ((int)(w/3) - 1)) == 0 || ((x) + ((int)(w % 3) - 1)) == 0 || ((y) + ((int)(w/3) - 1)) == surrSide + 1 || ((x) + ((int)(w % 3) - 1)) == surrSide + 1) ||
                                   (float)RGCdet_l[posY][posX].previousValues[tempParasolIndex] >= (float)perspLeft[currIndex + 3] ||
                                   RGCdet_l[posY][posX].perspX < 4 || RGCdet_l[posY][posX].perspY < 4 || RGCdet_l[posY][posX].perspX > rgcparams.perspWidth - 4 || RGCdet_l[posY][posX].perspY > rgcparams.perspHeight - 4){
                                    continue;
                                } else {
                                    // Check to make sure only greater shifts in weights are done
                                    if(RGCdet_l[posY][posX].parasolWeights_curr[tempParasolIndex] < (int)((((float)perspLeft[currIndex + 3] -
                                                                                                           (float)RGCdet_l[posY][posX].previousValues[tempParasolIndex]) / 255) * parasolWeight)){
                                        RGCdet_l[posY][posX].parasolWeights_curr[internalIndex] =
                                                (int)((((float)perspLeft[currIndex + 3] - (float)RGCdet_l[posY][posX].previousValues[internalIndex]) / 255) * parasolWeight);
                                    }
                                }
                            }
                        }
                    }
                }
                    // Midget - COLOR
                else if (RGCdet_l[posY][posX].detType == COLOR){
                    if(RGCdet_l[posY][posX].colID == R_rgc){
                        // Center is (S+L)                                                     // S                              // L
                        cenSumL += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspLeft[currIndex + 2] : (float)perspLeft[currIndex]));
                        cenIdeal += (xComp + yComp) * 255;
                    } else if (RGCdet_l[posY][posX].colID == G_rgc){
                        // Center is M                       // M
                        cenSumL += ((xComp + yComp) * (float)perspLeft[currIndex + 1]);
                        cenIdeal += (xComp + yComp) * 255;
                    } else if(RGCdet_l[posY][posX].colID == B_rgc){
                        // Center is (S+M)                                                     // S                              // M
                        cenSumL += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspLeft[currIndex + 2] : (float)perspLeft[currIndex + 1]));
                        cenIdeal += (xComp + yComp) * 255;
                    } else if (RGCdet_l[posY][posX].colID == Y_rgc){
                        // Center is L                       // L
                        cenSumL += ((xComp + yComp) * (float)perspLeft[currIndex]);
                        cenIdeal += (xComp + yComp) * 255;
                    }
                }
            }
            // ----------------------------- X -------------------------------
        }
    }
// The present becomes the past
    if(RGCdet_l[posY][posX].type == PARASOL && RGCdet_l[posY][posX].detType == MOTION){
        for (int y = 1; y < (surrSide) + 1; y+=1){
            // Y boundary condition
            if ((((RGCdet_l[posY][posX].perspY < midY) && (y < midY) && ((midY - y) > RGCdet_l[posY][posX].perspY)) ||
                 ((((rgcparams.perspHeight - 1) - (RGCdet_l[posY][posX].perspY)) < midY) && (y > midY) &&
                  ((y - midY) > ((rgcparams.perspHeight - 1) - (RGCdet_l[posY][posX].perspY)))))) {
                continue;
            }
            for(int x = 1; x < (surrSide) + 1; x += 1){
                // X boundary condition
                if ((((RGCdet_l[posY][posX].perspX < midX) && (x < midX) && ((midX - x) > RGCdet_l[posY][posX].perspX)) ||
                     ((((rgcparams.perspWidth - 1) - (RGCdet_l[posY][posX].perspX)) < midX) && (x > midX) &&
                      ((x - midX) > ((rgcparams.perspWidth - 1) - (RGCdet_l[posY][posX].perspX)))))) {
                    continue;
                }
                internalIndex = ((y - 1) * surrSide) + (x - 1);
                currIndex = cenIndex + ((y - midY) * 4 * rgcparams.perspWidth) + ((x - midX) * 4);
                RGCdet_l[posY][posX].previousValues[internalIndex] = (float)perspLeft[currIndex + 3];
                if(RGCdet_l[posY][posX].parasolWeights_curr[internalIndex] > 0) RGCdet_l[posY][posX].parasolWeights_curr[internalIndex] -= 1;
            }
        }
    }
    // Calculating surround averages
    surrValL = surrSumL / surrIdeal;
    // Calculating center averages
    cenValL = cenSumL / cenIdeal;

    if(RGCdet_l[posY][posX].type == MIDGET && RGCdet_l[posY][posX].detType == LUM){
        perspTest[RGCdet_l[posY][posX].perspID] =  (int)(255 * ((((cenValL - surrValL) < 0) ? -0 : 1) * 15 *(cenValL - surrValL)));
        perspTest[RGCdet_l[posY][posX].perspID + 1] = (int)(255 * ((((cenValL - surrValL) < 0) ? -0 : 1) * 15 * (cenValL - surrValL)));
        perspTest[RGCdet_l[posY][posX].perspID + 2] =  (int)(255 * ((((cenValL - surrValL) < 0) ? -0 : 1) * 15 *(cenValL - surrValL)));
        perspTest[RGCdet_l[posY][posX].perspID + 3] = (int)(255 * ((((cenValL - surrValL) < 0) ? -0 : 1) * 15 *(cenValL - surrValL)));
    }
    if(RGCdet_l[posY][posX].type == MIDGET && RGCdet_l[posY][posX].detType == COLOR && RGCdet_l[posY][posX].colID == Y_rgc){
        perspTest[RGCdet_l[posY][posX].perspID] = (int)(255 * ((((cenValL - surrValL) < 0) ? -0 : 1) * 15 *(cenValL - surrValL)));
        perspTest[RGCdet_l[posY][posX].perspID + 1] = (int)(255 * ((((cenValL - surrValL) < 0) ? -0 : 1) * 15 *(cenValL - surrValL)));
        perspTest[RGCdet_l[posY][posX].perspID + 2] =  (int)0;
        perspTest[RGCdet_l[posY][posX].perspID + 3] = (int)(int)(255 * ((((cenValL - surrValL) < 0) ? -0 : 1) * 15 *(cenValL - surrValL)));
    }
    // To rectify -ve responses, just multiply the below with
    // (((cenValR - surrValR) < 0) ? -0 : 1)
    leftInputs[posY][posX] = RGCdet_l[posY][posX].perspX;

}

__global__
void formRGCcurrents_r(RGCPARAMS rgcparams, uint8_t *perspTest, uint8_t  *perspRight, float** rightInputs, int* xWidths, int* yMatch, RGC** RGCdet_r, int frameNum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Gets the current coords in the RGC array
    int tempCumWidth = 0, posX = 0, posY = 0;
    for(int j = 0; j < rgcparams.rgcArrLen; j+=1){
        tempCumWidth += xWidths[j];
        if(i < tempCumWidth){
            posY = j;
            posX = i - (tempCumWidth - xWidths[j]);
            break;
        }
    }

    float surrSumR = 0, cenSumR = 0, surrIdeal = 0, cenIdeal = 0, surrValR = 0, cenValR = 0,
            xComp, yComp;
    int midX, midY, surrSide = (RGCdet_r[posY][posX].cenRfSide + (2 * RGCdet_r[posY][posX].surRfWidth)), currIndex, cenIndex, testr = 0;
    if(surrSide % 2 == 0) {
        midX = surrSide / 2;
        midY = midX;
    }
    if(surrSide % 2 == 1) {
        midX = (surrSide / 2) + 1;
        midY = midX;
    }

    cenIndex = (RGCdet_r[posY][posX].perspY * 4 * rgcparams.perspWidth) + (RGCdet_r[posY][posX].perspX * 4);

    for (int y = 1; y < (surrSide) + 1; y+=1){
        // Y boundary condition
        if ((((RGCdet_r[posY][posX].perspY < midY) && (y < midY) && ((midY - y) > RGCdet_r[posY][posX].perspY)) ||
             ((((rgcparams.perspHeight - 1) - (RGCdet_r[posY][posX].perspY)) < midY) && (y > midY) &&
              ((y - midY) > ((rgcparams.perspHeight - 1) - (RGCdet_r[posY][posX].perspY)))))) {
            continue;
        }
        for(int x = 1; x < (surrSide) + 1; x += 1){
            // X boundary condition
            if ((((RGCdet_r[posY][posX].perspX < midX) && (x < midX) && ((midX - x) > RGCdet_r[posY][posX].perspX)) ||
                 ((((rgcparams.perspWidth - 1) - (RGCdet_r[posY][posX].perspX)) < midX) && (x > midX) &&
                  ((x - midX) > ((rgcparams.perspWidth - 1) - (RGCdet_r[posY][posX].perspX)))))) {
                continue;
            }
            currIndex = cenIndex + ((y - midY) * 4 * rgcparams.perspWidth) + ((x - midX) * 4);
            xComp = 0; yComp = 0;

            // --------------------------- Surround Region -------------------------------
            if((x <= (RGCdet_r[posY][posX].surRfWidth)) || (x > (surrSide - RGCdet_r[posY][posX].surRfWidth)) ||
               (y <= (RGCdet_r[posY][posX].surRfWidth)) || (y > (surrSide - RGCdet_r[posY][posX].surRfWidth))){
                xComp = (float)x;
                yComp = (float)y;
                if(x > midX){
                    xComp = (float)((surrSide + 1) - x);
                }
                if(y > midY){
                    yComp = (float)((surrSide + 1) - y);
                }
                if (RGCdet_r[posY][posX].detType == LUM){
                    testr+=1;
                    surrSumR += (float)(xComp + yComp) * (float)perspRight[currIndex + 3];
                    surrIdeal += (float)(xComp + yComp) * 255;
                } else if (RGCdet_r[posY][posX].detType == COLOR){
                    if(RGCdet_r[posY][posX].colID == R_rgc){
                        // Surround is -M                     // M
                        surrSumR += ((xComp + yComp) * (float)perspRight[currIndex + 1]);
                        surrIdeal += (xComp + yComp) * 255;
                    } else if (RGCdet_r[posY][posX].colID == G_rgc){
                        // Surround is -(S + L)                                                 // S                              // L
                        surrSumR += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspRight[currIndex + 2] : (float)perspRight[currIndex]));
                        surrIdeal += (xComp + yComp) * 255;
                    } else if(RGCdet_r[posY][posX].colID == B_rgc){
                        // Surround is -L                     // L
                        surrSumR += ((xComp + yComp) * (float)perspRight[currIndex]);
                        surrIdeal += (xComp + yComp) * 255;
                    } else if (RGCdet_r[posY][posX].colID == Y_rgc){
                        // Surround is -(S + M)                                                 // S                              // M
                        surrSumR += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspRight[currIndex + 2] : (float)perspRight[currIndex + 1]));
                        surrIdeal += (xComp + yComp) * 255;
                    }
                }
            }
                // ----------------------------- X -------------------------------
                // --------------------------- Center Region -------------------------------
            else {
                xComp = (float)x - RGCdet_r[posY][posX].surRfWidth;
                yComp = (float)y - RGCdet_r[posY][posX].surRfWidth;
                if(x > midX){
                    xComp = (float)((RGCdet_r[posY][posX].cenRfSide + RGCdet_r[posY][posX].surRfWidth + 1) - x);
                }
                if(y > midY){
                    yComp = (float)((RGCdet_r[posY][posX].cenRfSide + RGCdet_r[posY][posX].surRfWidth + 1) - y);
                }
                if (RGCdet_r[posY][posX].detType == LUM){

                    cenSumR += (float)(xComp + yComp) * (float)perspRight[currIndex + 3];
                    cenIdeal += (float)(xComp + yComp) * 255;
                } else if (RGCdet_r[posY][posX].detType == COLOR){
                    if(RGCdet_r[posY][posX].colID == R_rgc){
                        // Center is (S+L)                                                     // S                              // L
                        cenSumR += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspRight[currIndex + 2] : (float)perspRight[currIndex]));
                        cenIdeal += (xComp + yComp) * 255;
                    } else if (RGCdet_r[posY][posX].colID == G_rgc){
                        // Center is M                       // M
                        cenSumR += ((xComp + yComp) * (float)perspRight[currIndex + 1]);
                        cenIdeal += (xComp + yComp) * 255;
                    } else if(RGCdet_r[posY][posX].colID == B_rgc){
                        // Center is (S+M)                                                     // S                              // M
                        cenSumR += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspRight[currIndex + 2] : (float)perspRight[currIndex + 1]));
                        cenIdeal += (xComp + yComp) * 255;
                    } else if (RGCdet_r[posY][posX].colID == Y_rgc){
                        // Center is L                       // L
                        cenSumR += ((xComp + yComp) * (float)perspRight[currIndex]);
                        cenIdeal += (xComp + yComp) * 255;
                    }
                }
            }
            // ----------------------------- X -------------------------------
        }
    }
    // Calculating surround averages
    surrValR = surrSumR / surrIdeal;
    // Calculating center averages
    cenValR = cenSumR / cenIdeal;


//    if(RGCdet_r[posY][posX].type == MIDGET){
//        perspTest[RGCdet_r[posY][posX].perspID] =  (int)(255);
//        perspTest[RGCdet_r[posY][posX].perspID + 1] = (int)(255);
//        perspTest[RGCdet_r[posY][posX].perspID + 2] =  (int)(255);
//        perspTest[RGCdet_r[posY][posX].perspID + 3] = (int)(255);
//    }
    // To rectify -ve responses, just multiply the below with
    // (((cenValR - surrValR) < 0) ? -0 : 1)
    rightInputs[posY][posX] = RGCdet_r[posY][posX].perspX;
}

__global__
void ffmpeg2World(::uint8_t  *worldLeft, ::uint8_t  *worldRight, ::uint8_t  *ffly, ::uint8_t  *ffry, ::uint8_t  *fflu, ::uint8_t  *ffru, ::uint8_t  *fflv, ::uint8_t  *ffrv, PARAMS devParams){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int u = i * 4;
    int p_y = i;

    int corrID = (((i / devParams.worldWidth)/2) * devParams.vLineSize) + ((i % devParams.worldWidth) / 2);

    // R
    worldLeft[u] = ffly[p_y] + (1.370705 * (fflv[corrID] - 128));
    worldRight[u] = ffry[p_y] + (1.370705 * (ffrv[corrID] - 128));

    // G
    worldLeft[u + 1] = ffly[p_y] - (0.337633 * (fflu[corrID] - 128)) - (0.698001 * (fflv[corrID] - 128));
    worldRight[u + 1] = ffry[p_y] - (0.337633 * (ffru[corrID] - 128)) - (0.698001 * (ffrv[corrID] - 128));

    // B
    worldLeft[u + 2] = ffly[p_y] + 1.732446 * (fflu[corrID] - 128);
    worldRight[u + 2] = ffry[p_y] + 1.732446 * (ffru[corrID] - 128);

    // A
    worldLeft[u + 3] = ffly[p_y];
    worldRight[u + 3] = ffry[p_y];

}

__global__
void ffmpeg2Persp(::uint8_t  *perspLeft, ::uint8_t  *perspRight, ::uint8_t  *ffly, ::uint8_t  *ffry, ::uint8_t  *fflu, ::uint8_t  *ffru, ::uint8_t  *fflv, ::uint8_t  *ffrv, PARAMS devParams){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int u = i * 4;
    int p_y = i;

    int corrID = (((i / devParams.perspWidth)/2) * devParams.vLineSize) + ((i % devParams.perspWidth) / 2);

    // R
    perspLeft[u] = ffly[p_y] + (1.370705 * (fflv[corrID] - 128));
    perspRight[u] = ffry[p_y] + (1.370705 * (ffrv[corrID] - 128));

    // G
    perspLeft[u + 1] = ffly[p_y] - (0.337633 * (fflu[corrID] - 128)) - (0.698001 * (fflv[corrID] - 128));
    perspRight[u + 1] = ffry[p_y] - (0.337633 * (ffru[corrID] - 128)) - (0.698001 * (ffrv[corrID] - 128));

    // B
    perspLeft[u + 2] = ffly[p_y] + 1.732446 * (fflu[corrID] - 128);
    perspRight[u + 2] = ffry[p_y] + 1.732446 * (ffru[corrID] - 128);

    // A
    perspLeft[u + 3] = ffly[p_y];
    perspRight[u + 3] = ffry[p_y];

}

void vid2rgc (int* frameNum, queue<float**> *rgcQueue_l, queue<float**> *rgcQueue_r, queue<RGC**> *rgcDetsQ, vid2rgcParams *v2rp, mutex &rgcMut, condition_variable &rgcCond){


    // Video processing parameters
    VideoReaderState vr_stateLeft, vr_stateRight;
    if (!video_reader_open(&vr_stateLeft, "/home/kasm-user/CLionProjects/BlackTest/src/cud/walk.mp4")) {
        cout << "ERROR!!" << endl;
        cout << "Couldn't open video file (make sure you set a video file that exists" << endl;
    }
    if (!video_reader_open(&vr_stateRight, "/home/kasm-user/CLionProjects/BlackTest/src/cud/walk.mp4")) {
        cout << "ERROR!!" << endl;
        cout << "Couldn't open video file (make sure you set a video file that exists" << endl;
    }

    int inputMode  = VID;

    // Allocate frameLeft buffer
    constexpr int ALIGNMENT = 128;
    const int frame_width = vr_stateLeft.width;
    const int frame_height = vr_stateLeft.height;
    const int perspHeight = 2160;
    const int perspWidth = 3840;
    const int camWidth = 1920;
    const int camHeight = 1080;
    int numOfPixels = frame_width * frame_height;
    int perspNumofPixels = perspWidth * perspHeight;
    int *retinaDivs;

    GLFWwindow* window;

    if (!glfwInit()) {
        printf("Couldn't init GLFW\n");
    }

    window = glfwCreateWindow(camWidth, camHeight, "Hello World", NULL, NULL);
    if (!window) {
        printf("Couldn't open window\n");
    }

    glfwMakeContextCurrent(window);

    // Generate texture
    GLuint tex_handle;
    glGenTextures(1, &tex_handle);
    glBindTexture(GL_TEXTURE_2D, tex_handle);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up orphographic projection
    int window_width, window_height;
    glfwGetFramebufferSize(window, &window_width, &window_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, window_width, window_height, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);


    // Defining perspective windows

    rgcparams.divFactors = (int*)malloc(6 * sizeof(int));

    if (perspWidth > 5000) {
        cout << "Resolution of Perspective is 8K" << endl;
        rgcparams.foveaWidth = 300;
        rgcparams.divFactors[0] = 1;
        rgcparams.paraLength = 200;
        rgcparams.divFactors[1] = 2;
        rgcparams.periLength = 300;
        rgcparams.divFactors[2] = 3;
        rgcparams.oz1up = 400;
        rgcparams.oz1side = 500;
        rgcparams.divFactors[3] = 4;
        rgcparams.oz2up = 500;
        rgcparams.oz2side = 1000;
        rgcparams.divFactors[4] = 8;
        rgcparams.oz3up = 600;
        rgcparams.oz3side = 1500;
        rgcparams.divFactors[5] = 12;
    }
    else if (perspWidth > 3000 && perspWidth < 5000){
        cout << "Resolution of Perspective is 4K" << endl; //4145280 - foveal point
        rgcparams.foveaWidth = 90;
        rgcparams.divFactors[0] = 1; // 32,400 - 129,600
        rgcparams.paraLength = 120;
        rgcparams.divFactors[1] = 2; // 45,000 - 90,000
        rgcparams.periLength = 180;
        rgcparams.divFactors[2] = 3; // 34,000
        rgcparams.oz1up = 200;
        rgcparams.oz1side = 300;
        rgcparams.divFactors[3] = 4; // 60,000 122400-corners 336600-sides 158400-tops
        rgcparams.oz2up = 300;
        rgcparams.oz2side = 510;
        rgcparams.divFactors[4] = 5; // 81,000
        rgcparams.oz3up = 200;
        rgcparams.oz3side = 765;
        rgcparams.divFactors[5] = 7; // 83,686
    }

    rgcparams.perspHeight = perspHeight;
    rgcparams.perspWidth = perspWidth;
    int tillPara = rgcparams.foveaWidth + rgcparams.paraLength,
            tillPeri = tillPara + rgcparams.periLength,
            tillOZ1 = tillPeri + rgcparams.oz1up,
            tillOZ2 = tillOZ1 + rgcparams.oz2up,
            tillOZ3 = tillOZ2 + rgcparams.oz3up,
            OZ3ellipse = tillOZ3 + 1000,
            OZ2ellipse = tillOZ2 + 600,

            fovCenSide_m = 2, fovSurrSide_m = 5, fovCenSide_p = 6, fovSurrSide_p = 8,
            paraCenSide_m = 3, paraSurrSide_m = 10, paraCenSide_p = 8, paraSurrSide_p = 10,
            periCenSide_m = 6, periSurrSide_m = 16, periCenSide_p = 12, periSurrSide_p = 14,
            oz1CenSide_m = 8, oz1SurrSide_m = 20, oz1CenSide_p = 15, oz1SurrSide_p = 18,
            oz2CenSide_m = 14, oz2SurrSide_m = 40, oz2CenSide_p = 24, oz2SurrSide_p = 24,
            oz3CenSide_m = 20, oz3SurrSide_m = 60, oz3CenSide_p = 36, oz3SurrSide_p = 48,
            tmpCenSide_m = 0, tmpSurrSide_m = 0, tmpCenSide_p = 0, tmpSurrSide_p = 0;

    //------ Init - VAriables required to calculate RGC inputs in initRGCdets -----------
    int *yMatchHost, *xWidthsHost, *yMatchDev, *xWidthsDev, tmpxSum, rgcArrayHeight = -1, RGCcount = 0, rgcInd = 0,
            fovy = ((perspHeight / 2) - 1), fovx = ((perspWidth / 2) - 1), diffX = 0, diffY = 0, divR = 0, mostProb = 0,
            ElXGr = 0, ElYGr = 0, ElXLs = 0, ElYLs = 0, divSector_df = 0, divSector_mrfCen = 0, divSector_prfCen = 0,
            divSector_mrfSurr = 0, divSector_prfSurr = 0;
    RGC swapVar;
    double SOL = 0, SOR = 0, eSOL = 0, SOLGr = 0, SOLLs = 0;
    random_device rd;     // Only used once to initialise (seed) engine
    mt19937 rng(rd());    // Random-number engine used (Mersenne-Twister in this case)
    uniform_real_distribution<float> uni(0,1); // Guaranteed unbiased
    float rnd = uni(rng),
    // The random number that's reset throughout each relevant scope
    currRand = 0,
    // How much the radius has progressed between two zones
    radProg = 0,
    // The individual lengths of areas for each of the divFactors under the probability graph
    divLength_df = 0, divLength_mrfCen = 0, divLength_prfCen = 0, divLength_mrfSurr = 0, divLength_prfSurr = 0,
    // angle between point and origin
    angle = 0, angleDeg = 0;
    RGC** tmpRGCdets_l = (RGC**) malloc(perspHeight * sizeof(RGC*)),
            **tmpRGCdets_r = (RGC**) malloc(perspHeight * sizeof(RGC*));
    yMatchHost = (int *)malloc(perspHeight * sizeof(int));
    xWidthsHost = (int *)malloc(perspHeight * sizeof(int));
    //------------------------------------------------------------------
    for(int i = 0; i < perspHeight; i+=1){
        tmpxSum = 0;
        for(int j = 0; j < perspWidth; j+=1){
            diffX = fovx - j; diffY = fovy - i;
            currRand = uni(rng);
            SOL = pow(diffX, 2) + pow(diffY, 2);

            // ---------------------- FOVEA ---------------------------
            if(SOL < pow(rgcparams.foveaWidth, 2) && (i % rgcparams.divFactors[0] == 0) && (j % rgcparams.divFactors[0] == 0)){
                if(tmpxSum == 0){
                    rgcArrayHeight+=1;
                    tmpRGCdets_l[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    tmpRGCdets_r[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                }
                tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                currRand = uni(rng);
                if(currRand > 0.95){
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = PARASOL;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = LUM;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = fovCenSide_p;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = fovSurrSide_p;
                } else {
                    currRand = uni(rng);
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = MIDGET;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = LUM;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = fovCenSide_m;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = fovSurrSide_m;
                    currRand = uni(rng);
                    if(currRand < 0.25) {
                        currRand = uni(rng);
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = COLOR;
                        if(currRand < 0.4) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = R_rgc;
                        if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = G_rgc;
                        if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = B_rgc;
                        if(currRand >= 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                    }
                }
                tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspX = j;
                tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspY = i;
                tmpRGCdets_l[rgcArrayHeight][tmpxSum].zone = FOV;

                // Right eye entry
                tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                currRand = uni(rng);
                if(currRand > 0.95){
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = PARASOL;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = LUM;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = fovCenSide_p;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = fovSurrSide_p;
                } else {
                    currRand = uni(rng);
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = MIDGET;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = LUM;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = fovCenSide_m;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = fovSurrSide_m;
                    currRand = uni(rng);
                    if(currRand < 0.25) {
                        currRand = uni(rng);
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = COLOR;
                        if(currRand < 0.4) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = R_rgc;
                        if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = G_rgc;
                        if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = B_rgc;
                        if(currRand >= 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                    }
                }
                tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspX = j;
                tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspY = i;
                tmpRGCdets_r[rgcArrayHeight][tmpxSum].zone = FOV;
                tmpxSum+=1;
            }
            // ----------------------- X --------------------------------

            // ---------------------- PARAFOVEAL REGION --------------------

            if(SOL < pow(tillPara, 2) && SOL >= pow(rgcparams.foveaWidth, 2)){

                // First, we find out just how far between the two bands the current radius is
                radProg = (((double)(SOL - pow(rgcparams.foveaWidth, 2)) / (double)(pow(tillPara, 2) - pow(rgcparams.foveaWidth, 2))));

                // Then we calculate the length of the subdivisions within the band
                divLength_df = 1 / (float)((rgcparams.divFactors[1] - rgcparams.divFactors[0]));
                divLength_mrfCen = 1 / (float)((paraCenSide_m - fovCenSide_m) > 0 ? (paraCenSide_m - fovCenSide_m) : -1);
                divLength_mrfSurr = 1 / (float)((paraSurrSide_m - fovSurrSide_m) > 0 ? (paraSurrSide_m - fovSurrSide_m) : -1);
                divLength_prfCen = 1 / (float)((paraCenSide_p - fovCenSide_p) > 0 ? (paraCenSide_p - fovCenSide_p) : -1);
                divLength_prfSurr = 1 / (float)((paraSurrSide_p - fovSurrSide_p) > 0 ? (paraSurrSide_p - fovSurrSide_p) : -1);

                // Calculation of respective RF sectors
                divSector_mrfCen = (int)(radProg / divLength_mrfCen);
                divSector_mrfSurr = (int)(radProg / divLength_mrfSurr);
                divSector_prfCen = (int)(radProg / divLength_prfCen);
                divSector_prfSurr = (int)(radProg / divLength_prfSurr);
                mostProb = (int)(radProg * (float)((rgcparams.divFactors[1] - rgcparams.divFactors[0]) + 1));

                currRand = uni(rng);
                // ------------ Midget Centre ------------
                if(divLength_mrfCen == -1){
                    tmpCenSide_m = paraCenSide_m;
                }else {
                    if(radProg > 1 - divLength_mrfCen){
                        if (currRand < ((radProg - (1 - divLength_mrfCen)) / divLength_mrfCen)) {
                            tmpCenSide_m = paraCenSide_m;
                        }
                        else {
                            tmpCenSide_m = paraCenSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfCen * divLength_mrfCen)) / divLength_mrfCen)){
                            tmpCenSide_m = fovCenSide_m + divSector_mrfCen;
                        } else {
                            tmpCenSide_m = fovCenSide_m + divSector_mrfCen + 1;
                        }
                    }
                }

                // ------------ Midget Surround------------
                if(divLength_mrfSurr == -1){
                    tmpSurrSide_m = paraSurrSide_m;
                } else {
                    if(radProg > 1 - divLength_mrfSurr){
                        if (currRand < ((radProg - (1 - divLength_mrfSurr)) / divLength_mrfSurr)) {
                            tmpSurrSide_m = paraSurrSide_m;
                        }
                        else {
                            tmpSurrSide_m = paraSurrSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfSurr * divLength_mrfSurr)) / divLength_mrfSurr)){
                            tmpSurrSide_m = fovSurrSide_m + divSector_mrfSurr;
                        } else {
                            tmpSurrSide_m = fovSurrSide_m + divSector_mrfSurr + 1;
                        }
                    }
                }

                // ------------ Parasol Centre ------------
                if(divLength_prfCen == -1){
                    tmpCenSide_p = paraCenSide_p;
                } else {
                    if(radProg > 1 - divLength_prfCen){
                        if (currRand < ((radProg - (1 - divLength_prfCen)) / divLength_prfCen)) {
                            tmpCenSide_p = paraCenSide_p;
                        }
                        else {
                            tmpCenSide_p = paraCenSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfCen * divLength_prfCen)) / divLength_prfCen)){
                            tmpCenSide_p = fovCenSide_p + divSector_prfCen;
                        } else {
                            tmpCenSide_p = fovCenSide_p + divSector_prfCen + 1;
                        }
                    }
                }

                // ------------ Parasol Surround ------------
                if(divLength_prfSurr == -1){
                    tmpSurrSide_p = paraSurrSide_p;
                } else {
                    if(radProg > 1 - divLength_prfSurr){
                        if (currRand < ((radProg - (1 - divLength_prfSurr)) / divLength_prfSurr)) {
                            tmpSurrSide_p = paraSurrSide_p;
                        }
                        else {
                            tmpSurrSide_p = paraSurrSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfSurr * divLength_prfSurr)) / divLength_prfSurr)){
                            tmpSurrSide_p = fovSurrSide_p + divSector_prfSurr;
                        } else {
                            tmpSurrSide_p = fovSurrSide_p + divSector_prfSurr + 1;
                        }
                    }
                }

                // ------------- For divFactors -------------
                // Last band
                if(radProg > 1 - divLength_df) {
                    if (currRand < ((radProg - (1 - divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[1];
                    }
                    else {
                        divR = rgcparams.divFactors[1] - 1;
                    }
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector_df = (int)(radProg / divLength_df);

                    // If currRand is greater than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector_df * divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[0] + divSector_df;
                    }

                        // if currRand is lesser
                    else {
                        divR = rgcparams.divFactors[0] + divSector_df + 1;
                    }
                }

                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets_l[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                        tmpRGCdets_r[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.95){
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].zone = PARA;

                    // Right eye
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.95){
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].zone = PARA;
                    tmpxSum+=1;
                }
            }
            // ----------------------- X --------------------------

            // ---------------------- PERIFOVEAL REGION --------------------

            if(SOL < pow(tillPeri, 2) && SOL >= pow(tillPara, 2)){

                // First, we find out just how far between the two bands the current radius is
                radProg = (((double)(SOL - pow(tillPara, 2)) / (double)(pow(tillPeri, 2) - pow(tillPara, 2))));

                // Then we calculate the length of the subdivisions within the band
                divLength_df = 1 / (float)((rgcparams.divFactors[2] - rgcparams.divFactors[1]));
                divLength_mrfCen = 1 / (float)((periCenSide_m - paraCenSide_m) > 0 ? (periCenSide_m - paraCenSide_m) : -1);
                divLength_mrfSurr = 1 / (float)((periSurrSide_m - paraSurrSide_m) > 0 ? (periSurrSide_m - paraSurrSide_m) : -1);
                divLength_prfCen = 1 / (float)((periCenSide_p - paraCenSide_p) > 0 ? (periCenSide_p - paraCenSide_p) : -1);
                divLength_prfSurr = 1 / (float)((periSurrSide_p - paraSurrSide_p) > 0 ? (periSurrSide_p - paraSurrSide_p) : -1);

                // Calculation of respective RF sectors
                divSector_mrfCen = (int)(radProg / divLength_mrfCen);
                divSector_mrfSurr = (int)(radProg / divLength_mrfSurr);
                divSector_prfCen = (int)(radProg / divLength_prfCen);
                divSector_prfSurr = (int)(radProg / divLength_prfSurr);

                currRand = uni(rng);
                // ------------ Midget Centre ------------
                if(divLength_mrfCen == -1){
                    tmpCenSide_m = periCenSide_m;
                }else {
                    if(radProg > 1 - divLength_mrfCen){
                        if (currRand < ((radProg - (1 - divLength_mrfCen)) / divLength_mrfCen)) {
                            tmpCenSide_m = periCenSide_m;
                        }
                        else {
                            tmpCenSide_m = periCenSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfCen * divLength_mrfCen)) / divLength_mrfCen)){
                            tmpCenSide_m = paraCenSide_m + divSector_mrfCen;
                        } else {
                            tmpCenSide_m = paraCenSide_m + divSector_mrfCen + 1;
                        }
                    }
                }

                // ------------ Midget Surround------------
                if(divLength_mrfSurr == -1){
                    tmpSurrSide_m = periSurrSide_m;
                } else {
                    if(radProg > 1 - divLength_mrfSurr){
                        if (currRand < ((radProg - (1 - divLength_mrfSurr)) / divLength_mrfSurr)) {
                            tmpSurrSide_m = periSurrSide_m;
                        }
                        else {
                            tmpSurrSide_m = periSurrSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfSurr * divLength_mrfSurr)) / divLength_mrfSurr)){
                            tmpSurrSide_m = paraSurrSide_m + divSector_mrfSurr;
                        } else {
                            tmpSurrSide_m = paraSurrSide_m + divSector_mrfSurr + 1;
                        }
                    }
                }

                // ------------ Parasol Centre ------------
                if(divLength_prfCen == -1){
                    tmpCenSide_p = periCenSide_p;
                } else {
                    if(radProg > 1 - divLength_prfCen){
                        if (currRand < ((radProg - (1 - divLength_prfCen)) / divLength_prfCen)) {
                            tmpCenSide_p = periCenSide_p;
                        }
                        else {
                            tmpCenSide_p = periCenSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfCen * divLength_prfCen)) / divLength_prfCen)){
                            tmpCenSide_p = paraCenSide_p + divSector_prfCen;
                        } else {
                            tmpCenSide_p = paraCenSide_p + divSector_prfCen + 1;
                        }
                    }
                }

                // ------------ Parasol Surround ------------
                if(divLength_prfSurr == -1){
                    tmpSurrSide_p = periSurrSide_p;
                } else {
                    if(radProg > 1 - divLength_prfSurr){
                        if (currRand < ((radProg - (1 - divLength_prfSurr)) / divLength_prfSurr)) {
                            tmpSurrSide_p = periSurrSide_p;
                        }
                        else {
                            tmpSurrSide_p = periSurrSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfSurr * divLength_prfSurr)) / divLength_prfSurr)){
                            tmpSurrSide_p = paraSurrSide_p + divSector_prfSurr;
                        } else {
                            tmpSurrSide_p = paraSurrSide_p + divSector_prfSurr + 1;
                        }
                    }
                }

                // If it's the last band
                if(radProg > 1 - divLength_df) {
                    if (currRand < ((radProg - (1 - divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[2];
                    }
                    else {
                        divR = rgcparams.divFactors[2] - 1;
                    }
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector_df = (int)(radProg / divLength_df);

                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector_df * divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[1] + divSector_df;
                    }

                        // if currRand is greater
                    else {
                        divR = rgcparams.divFactors[1] + divSector_df + 1;
                    }
                }
                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets_l[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                        tmpRGCdets_r[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.9){
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].zone = PERI;

                    // Right eye
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.9){
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].zone = PERI;
                    tmpxSum+=1;
                }

            }
            // ----------------------- X --------------------------

            // ---------------------- OZ1 REGION --------------------

            if(SOL < pow(tillOZ1, 2) && SOL >= pow(tillPeri, 2)){

                // First, we find out just how far between the two bands the current radius is
                radProg = (((double)(SOL - pow(tillPeri, 2)) / (double)(pow(tillOZ1, 2) - pow(tillPeri, 2))));

                // Then we calculate the length of the subdivisions within the band
                divLength_df = 1 / (float)((rgcparams.divFactors[3] - rgcparams.divFactors[2]));
                divLength_mrfCen = 1 / (float)((oz1CenSide_m - periCenSide_m) > 0 ? (oz1CenSide_m - periCenSide_m) : -1);
                divLength_mrfSurr = 1 / (float)((oz1SurrSide_m - periSurrSide_m) > 0 ? (oz1SurrSide_m - periSurrSide_m) : -1);
                divLength_prfCen = 1 / (float)((oz1CenSide_p - periCenSide_p) > 0 ? (oz1CenSide_p - periCenSide_p) : -1);
                divLength_prfSurr = 1 / (float)((oz1SurrSide_p - periSurrSide_p) > 0 ? (oz1SurrSide_p - periSurrSide_p) : -1);

                // Calculation of respective RF sectors
                divSector_mrfCen = (int)(radProg / divLength_mrfCen);
                divSector_mrfSurr = (int)(radProg / divLength_mrfSurr);
                divSector_prfCen = (int)(radProg / divLength_prfCen);
                divSector_prfSurr = (int)(radProg / divLength_prfSurr);

                currRand = uni(rng);
                // ------------ Midget Centre ------------
                if(divLength_mrfCen == -1){
                    tmpCenSide_m = oz1CenSide_m;
                }else {
                    if(radProg > 1 - divLength_mrfCen){
                        if (currRand < ((radProg - (1 - divLength_mrfCen)) / divLength_mrfCen)) {
                            tmpCenSide_m = oz1CenSide_m;
                        }
                        else {
                            tmpCenSide_m = oz1CenSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfCen * divLength_mrfCen)) / divLength_mrfCen)){
                            tmpCenSide_m = periCenSide_m + divSector_mrfCen;
                        } else {
                            tmpCenSide_m = periCenSide_m + divSector_mrfCen + 1;
                        }
                    }
                }

                // ------------ Midget Surround------------
                if(divLength_mrfSurr == -1){
                    tmpSurrSide_m = oz1SurrSide_m;
                } else {
                    if(radProg > 1 - divLength_mrfSurr){
                        if (currRand < ((radProg - (1 - divLength_mrfSurr)) / divLength_mrfSurr)) {
                            tmpSurrSide_m = oz1SurrSide_m;
                        }
                        else {
                            tmpSurrSide_m = oz1SurrSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfSurr * divLength_mrfSurr)) / divLength_mrfSurr)){
                            tmpSurrSide_m = periSurrSide_m + divSector_mrfSurr;
                        } else {
                            tmpSurrSide_m = periSurrSide_m + divSector_mrfSurr + 1;
                        }
                    }
                }

                // ------------ Parasol Centre ------------
                if(divLength_prfCen == -1){
                    tmpCenSide_p = oz1CenSide_p;
                } else {
                    if(radProg > 1 - divLength_prfCen){
                        if (currRand < ((radProg - (1 - divLength_prfCen)) / divLength_prfCen)) {
                            tmpCenSide_p = oz1CenSide_p;
                        }
                        else {
                            tmpCenSide_p = oz1CenSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfCen * divLength_prfCen)) / divLength_prfCen)){
                            tmpCenSide_p = periCenSide_p + divSector_prfCen;
                        } else {
                            tmpCenSide_p = periCenSide_p + divSector_prfCen + 1;
                        }
                    }
                }

                // ------------ Parasol Surround ------------
                if(divLength_prfSurr == -1){
                    tmpSurrSide_p = oz1SurrSide_p;
                } else {
                    if(radProg > 1 - divLength_prfSurr){
                        if (currRand < ((radProg - (1 - divLength_prfSurr)) / divLength_prfSurr)) {
                            tmpSurrSide_p = oz1SurrSide_p;
                        }
                        else {
                            tmpSurrSide_p = oz1SurrSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfSurr * divLength_prfSurr)) / divLength_prfSurr)){
                            tmpSurrSide_p = periSurrSide_p + divSector_prfSurr;
                        } else {
                            tmpSurrSide_p = periSurrSide_p + divSector_prfSurr + 1;
                        }
                    }
                }

                // If it's the last band
                if(radProg > 1 - divLength_df) {
                    if (currRand < ((radProg - (1 - divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[3];
                    }
                    else {
                        divR = rgcparams.divFactors[3] - 1;
                    }
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector_df = (int)(radProg / divLength_df);

                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector_df * divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[2] + divSector_df;
                    }

                        // if currRand is greater
                    else {
                        divR = rgcparams.divFactors[2] + divSector_df + 1;
                    }
                }

                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets_l[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                        tmpRGCdets_r[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.8){
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].zone = OZ1;

                    // Right eye
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.8){
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].zone = OZ1;
                    tmpxSum+=1;
                }

            }
            // ----------------------- X --------------------------

            // ---------------------- OZ2 REGION - Circular Side  --------------------

            if((SOL < pow(tillOZ2, 2) && (diffX < 0)) && SOL >= pow(tillOZ1, 2)){

                // First, we find out just how far between the two bands the current radius is
                radProg = (((double)(SOL - pow(tillOZ1, 2)) / (double)(pow(tillOZ2, 2) - pow(tillOZ1, 2))));

                // Then we calculate the length of the subdivisions within the band
                divLength_df = 1 / (float)((rgcparams.divFactors[4] - rgcparams.divFactors[3]));
                divLength_mrfCen = 1 / (float)((oz2CenSide_m - oz1CenSide_m) > 0 ? (oz2CenSide_m - oz1CenSide_m) : -1);
                divLength_mrfSurr = 1 / (float)((oz2SurrSide_m - oz1SurrSide_m) > 0 ? (oz2SurrSide_m - oz1SurrSide_m) : -1);
                divLength_prfCen = 1 / (float)((oz2CenSide_p - oz1CenSide_p) > 0 ? (oz2CenSide_p - oz1CenSide_p) : -1);
                divLength_prfSurr = 1 / (float)((oz2SurrSide_p - oz1SurrSide_p) > 0 ? (oz2SurrSide_p - oz1SurrSide_p) : -1);

                // Calculation of respective RF sectors
                divSector_mrfCen = (int)(radProg / divLength_mrfCen);
                divSector_mrfSurr = (int)(radProg / divLength_mrfSurr);
                divSector_prfCen = (int)(radProg / divLength_prfCen);
                divSector_prfSurr = (int)(radProg / divLength_prfSurr);

                currRand = uni(rng);
                // ------------ Midget Centre ------------
                if(divLength_mrfCen == -1){
                    tmpCenSide_m = oz2CenSide_m;
                }else {
                    if(radProg > 1 - divLength_mrfCen){
                        if (currRand < ((radProg - (1 - divLength_mrfCen)) / divLength_mrfCen)) {
                            tmpCenSide_m = oz2CenSide_m;
                        }
                        else {
                            tmpCenSide_m = oz2CenSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfCen * divLength_mrfCen)) / divLength_mrfCen)){
                            tmpCenSide_m = oz1CenSide_m + divSector_mrfCen;
                        } else {
                            tmpCenSide_m = oz1CenSide_m + divSector_mrfCen + 1;
                        }
                    }
                }

                // ------------ Midget Surround------------
                if(divLength_mrfSurr == -1){
                    tmpSurrSide_m = oz2SurrSide_m;
                } else {
                    if(radProg > 1 - divLength_mrfSurr){
                        if (currRand < ((radProg - (1 - divLength_mrfSurr)) / divLength_mrfSurr)) {
                            tmpSurrSide_m = oz2SurrSide_m;
                        }
                        else {
                            tmpSurrSide_m = oz2SurrSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfSurr * divLength_mrfSurr)) / divLength_mrfSurr)){
                            tmpSurrSide_m = oz1SurrSide_m + divSector_mrfSurr;
                        } else {
                            tmpSurrSide_m = oz1SurrSide_m + divSector_mrfSurr + 1;
                        }
                    }
                }

                // ------------ Parasol Centre ------------
                if(divLength_prfCen == -1){
                    tmpCenSide_p = oz2CenSide_p;
                } else {
                    if(radProg > 1 - divLength_prfCen){
                        if (currRand < ((radProg - (1 - divLength_prfCen)) / divLength_prfCen)) {
                            tmpCenSide_p = oz2CenSide_p;
                        }
                        else {
                            tmpCenSide_p = oz2CenSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfCen * divLength_prfCen)) / divLength_prfCen)){
                            tmpCenSide_p = oz1CenSide_p + divSector_prfCen;
                        } else {
                            tmpCenSide_p = oz1CenSide_p + divSector_prfCen + 1;
                        }
                    }
                }

                // ------------ Parasol Surround ------------
                if(divLength_prfSurr == -1){
                    tmpSurrSide_p = oz2SurrSide_p;
                } else {
                    if(radProg > 1 - divLength_prfSurr){
                        if (currRand < ((radProg - (1 - divLength_prfSurr)) / divLength_prfSurr)) {
                            tmpSurrSide_p = oz2SurrSide_p;
                        }
                        else {
                            tmpSurrSide_p = oz2SurrSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfSurr * divLength_prfSurr)) / divLength_prfSurr)){
                            tmpSurrSide_p = oz1SurrSide_p + divSector_prfSurr;
                        } else {
                            tmpSurrSide_p = oz1SurrSide_p + divSector_prfSurr + 1;
                        }
                    }
                }

                // If it's the last band
                if(radProg > 1 - divLength_df) {
                    if (currRand < ((radProg - (1 - divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[4];
                    }
                    else {
                        divR = rgcparams.divFactors[4] - 1;
                    }
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector_df = (int)(radProg / divLength_df);

                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector_df * divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[3] + divSector_df;
                    }

                        // if currRand is greater
                    else {
                        divR = rgcparams.divFactors[3] + divSector_df + 1;
                    }
                }
                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets_l[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                        tmpRGCdets_r[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.7){
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].zone = OZ2;

                    // Right eye
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.7){
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].zone = OZ2;
                    tmpxSum+=1;
                }

            }
            // ----------------------- X --------------------------

            // ---------------------- OZ2 REGION - Elliptical Side  --------------------

            if((((((float)pow(diffX, 2) / (float)pow(OZ2ellipse, 2)) + ((float)pow(diffY, 2) / (float)pow(tillOZ2, 2))) <= 1) && (diffX > 0)) && SOL >= pow(tillOZ1, 2)){

                angle = atan2(diffY, (j - fovx));
                angleDeg = angle * (180 / M_PI) > 0 ? angle * (180 / M_PI) : 360 - abs(angle * (180 / M_PI));
                ElXLs = ((-1 * tillOZ1 * tillOZ1) / (sqrt(pow(tillOZ1, 2) + (pow(tillOZ1, 2) * pow(tan(angle), 2)))));
                ElYLs = (((angleDeg <= 180 ? 1 : -1) * tillOZ1 * tillOZ1) / (sqrt(pow(tillOZ1, 2) + (pow(tillOZ1, 2) / pow(tan(angle), 2)))));
                ElXGr = ((-1 * tillOZ2 * OZ2ellipse) / (sqrt(pow(tillOZ2, 2) + (pow(OZ2ellipse, 2) * pow(tan(angle), 2)))));
                ElYGr = (((angleDeg <= 180 ? 1 : -1) * tillOZ2 * OZ2ellipse) / (sqrt(pow(OZ2ellipse, 2) + (pow(tillOZ2, 2) / pow(tan(angle), 2)))));
                SOLGr = pow(ElXGr, 2) + pow(ElYGr, 2);
                SOLLs = pow(ElXLs, 2) + pow(ElYLs, 2);
                // First, we find out just how far between the two bands the current radius is
                radProg = ((double)((SOL) - SOLLs) / (double)(SOLGr - SOLLs));

                // Then we calculate the length of the subdivisions within the band
                divLength_df = 1 / (float)((rgcparams.divFactors[4] - rgcparams.divFactors[3]));
                divLength_mrfCen = 1 / (float)((oz2CenSide_m - oz1CenSide_m) > 0 ? (oz2CenSide_m - oz1CenSide_m) : -1);
                divLength_mrfSurr = 1 / (float)((oz2SurrSide_m - oz1SurrSide_m) > 0 ? (oz2SurrSide_m - oz1SurrSide_m) : -1);
                divLength_prfCen = 1 / (float)((oz2CenSide_p - oz1CenSide_p) > 0 ? (oz2CenSide_p - oz1CenSide_p) : -1);
                divLength_prfSurr = 1 / (float)((oz2SurrSide_p - oz1SurrSide_p) > 0 ? (oz2SurrSide_p - oz1SurrSide_p) : -1);

                // Calculation of respective RF sectors
                divSector_mrfCen = (int)(radProg / divLength_mrfCen);
                divSector_mrfSurr = (int)(radProg / divLength_mrfSurr);
                divSector_prfCen = (int)(radProg / divLength_prfCen);
                divSector_prfSurr = (int)(radProg / divLength_prfSurr);

                currRand = uni(rng);
                // ------------ Midget Centre ------------
                if(divLength_mrfCen == -1){
                    tmpCenSide_m = oz2CenSide_m;
                }else {
                    if(radProg > 1 - divLength_mrfCen){
                        if (currRand < ((radProg - (1 - divLength_mrfCen)) / divLength_mrfCen)) {
                            tmpCenSide_m = oz2CenSide_m;
                        }
                        else {
                            tmpCenSide_m = oz2CenSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfCen * divLength_mrfCen)) / divLength_mrfCen)){
                            tmpCenSide_m = oz1CenSide_m + divSector_mrfCen;
                        } else {
                            tmpCenSide_m = oz1CenSide_m + divSector_mrfCen + 1;
                        }
                    }
                }

                // ------------ Midget Surround------------
                if(divLength_mrfSurr == -1){
                    tmpSurrSide_m = oz2SurrSide_m;
                } else {
                    if(radProg > 1 - divLength_mrfSurr){
                        if (currRand < ((radProg - (1 - divLength_mrfSurr)) / divLength_mrfSurr)) {
                            tmpSurrSide_m = oz2SurrSide_m;
                        }
                        else {
                            tmpSurrSide_m = oz2SurrSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfSurr * divLength_mrfSurr)) / divLength_mrfSurr)){
                            tmpSurrSide_m = oz1SurrSide_m + divSector_mrfSurr;
                        } else {
                            tmpSurrSide_m = oz1SurrSide_m + divSector_mrfSurr + 1;
                        }
                    }
                }

                // ------------ Parasol Centre ------------
                if(divLength_prfCen == -1){
                    tmpCenSide_p = oz2CenSide_p;
                } else {
                    if(radProg > 1 - divLength_prfCen){
                        if (currRand < ((radProg - (1 - divLength_prfCen)) / divLength_prfCen)) {
                            tmpCenSide_p = oz2CenSide_p;
                        }
                        else {
                            tmpCenSide_p = oz2CenSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfCen * divLength_prfCen)) / divLength_prfCen)){
                            tmpCenSide_p = oz1CenSide_p + divSector_prfCen;
                        } else {
                            tmpCenSide_p = oz1CenSide_p + divSector_prfCen + 1;
                        }
                    }
                }

                // ------------ Parasol Surround ------------
                if(divLength_prfSurr == -1){
                    tmpSurrSide_p = oz2SurrSide_p;
                } else {
                    if(radProg > 1 - divLength_prfSurr){
                        if (currRand < ((radProg - (1 - divLength_prfSurr)) / divLength_prfSurr)) {
                            tmpSurrSide_p = oz2SurrSide_p;
                        }
                        else {
                            tmpSurrSide_p = oz2SurrSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfSurr * divLength_prfSurr)) / divLength_prfSurr)){
                            tmpSurrSide_p = oz1SurrSide_p + divSector_prfSurr;
                        } else {
                            tmpSurrSide_p = oz1SurrSide_p + divSector_prfSurr + 1;
                        }
                    }
                }

                // If it's the last band
                if(radProg > 1 - divLength_df) {
                    divSector_df = (int)(radProg / divLength_df);
                    if (currRand < ((radProg - (1 - divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[4];
                    }
                    else {
                        divR = rgcparams.divFactors[4] - 1;
                    }
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector_df = (int)(radProg / divLength_df);

                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector_df * divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[3] + divSector_df;
                    }

                        // if currRand is greater
                    else {
                        divR = rgcparams.divFactors[3] + divSector_df + 1;
                    }
                }
                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets_l[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                        tmpRGCdets_r[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.7){
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].zone = OZ2;

                    // Right eye
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.7){
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].zone = OZ2;
                    tmpxSum+=1;
                }

            }
            // ----------------------- X --------------------------

            // ---------------------- OZ3 REGION - Circular side--------------------

            if(((SOL < pow(tillOZ3, 2)) && (diffX < 0)) && SOL >= pow(tillOZ2, 2)){

                // First, we find out just how far between the two bands the current radius is
                radProg = (((double)(SOL - pow(tillOZ2, 2)) / (double)(pow(tillOZ3, 2) - pow(tillOZ2, 2))));

                // Then we calculate the length of the subdivisions within the band
                divLength_df = 1 / (float)((rgcparams.divFactors[5] - rgcparams.divFactors[4]));
                divLength_mrfCen = 1 / (float)((oz3CenSide_m - oz2CenSide_m) > 0 ? (oz3CenSide_m - oz2CenSide_m) : -1);
                divLength_mrfSurr = 1 / (float)((oz3SurrSide_m - oz2SurrSide_m) > 0 ? (oz3SurrSide_m - oz2SurrSide_m) : -1);
                divLength_prfCen = 1 / (float)((oz3CenSide_p - oz2CenSide_p) > 0 ? (oz3CenSide_p - oz2CenSide_p) : -1);
                divLength_prfSurr = 1 / (float)((oz3SurrSide_p - oz2SurrSide_p) > 0 ? (oz3SurrSide_p - oz2SurrSide_p) : -1);

                // Calculation of respective RF sectors
                divSector_mrfCen = (int)(radProg / divLength_mrfCen);
                divSector_mrfSurr = (int)(radProg / divLength_mrfSurr);
                divSector_prfCen = (int)(radProg / divLength_prfCen);
                divSector_prfSurr = (int)(radProg / divLength_prfSurr);

                currRand = uni(rng);
                // ------------ Midget Centre ------------
                if(divLength_mrfCen == -1){
                    tmpCenSide_m = oz3CenSide_m;
                }else {
                    if(radProg > 1 - divLength_mrfCen){
                        if (currRand < ((radProg - (1 - divLength_mrfCen)) / divLength_mrfCen)) {
                            tmpCenSide_m = oz3CenSide_m;
                        }
                        else {
                            tmpCenSide_m = oz3CenSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfCen * divLength_mrfCen)) / divLength_mrfCen)){
                            tmpCenSide_m = oz2CenSide_m + divSector_mrfCen;
                        } else {
                            tmpCenSide_m = oz2CenSide_m + divSector_mrfCen + 1;
                        }
                    }
                }

                // ------------ Midget Surround------------
                if(divLength_mrfSurr == -1){
                    tmpSurrSide_m = oz3SurrSide_m;
                } else {
                    if(radProg > 1 - divLength_mrfSurr){
                        if (currRand < ((radProg - (1 - divLength_mrfSurr)) / divLength_mrfSurr)) {
                            tmpSurrSide_m = oz3SurrSide_m;
                        }
                        else {
                            tmpSurrSide_m = oz3SurrSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfSurr * divLength_mrfSurr)) / divLength_mrfSurr)){
                            tmpSurrSide_m = oz2SurrSide_m + divSector_mrfSurr;
                        } else {
                            tmpSurrSide_m = oz2SurrSide_m + divSector_mrfSurr + 1;
                        }
                    }
                }

                // ------------ Parasol Centre ------------
                if(divLength_prfCen == -1){
                    tmpCenSide_p = oz3CenSide_p;
                } else {
                    if(radProg > 1 - divLength_prfCen){
                        if (currRand < ((radProg - (1 - divLength_prfCen)) / divLength_prfCen)) {
                            tmpCenSide_p = oz3CenSide_p;
                        }
                        else {
                            tmpCenSide_p = oz3CenSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfCen * divLength_prfCen)) / divLength_prfCen)){
                            tmpCenSide_p = oz2CenSide_p + divSector_prfCen;
                        } else {
                            tmpCenSide_p = oz2CenSide_p + divSector_prfCen + 1;
                        }
                    }
                }

                // ------------ Parasol Surround ------------
                if(divLength_prfSurr == -1){
                    tmpSurrSide_p = oz3SurrSide_p;
                } else {
                    if(radProg > 1 - divLength_prfSurr){
                        if (currRand < ((radProg - (1 - divLength_prfSurr)) / divLength_prfSurr)) {
                            tmpSurrSide_p = oz3SurrSide_p;
                        }
                        else {
                            tmpSurrSide_p = oz3SurrSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfSurr * divLength_prfSurr)) / divLength_prfSurr)){
                            tmpSurrSide_p = oz2SurrSide_p + divSector_prfSurr;
                        } else {
                            tmpSurrSide_p = oz2SurrSide_p + divSector_prfSurr + 1;
                        }
                    }
                }

                // If it's the last band
                if(radProg > 1 - divLength_df) {
                    if (currRand < ((radProg - (1 - divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[5];
                    }
                    else {
                        divR = rgcparams.divFactors[5] - 1;
                    }
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector_df = (int)(radProg / divLength_df);

                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector_df * divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[4] + divSector_df;
                    }

                        // if currRand is greater
                    else {
                        divR = rgcparams.divFactors[4] + divSector_df + 1;
                    }
                }
                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets_l[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                        tmpRGCdets_r[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.6){
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].zone = OZ3;

                    // Right eye
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.6){
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].zone = OZ3;
                    tmpxSum+=1;
                }
            }
            // ----------------------- X --------------------------

            // ---------------------- OZ3 REGION - Elliptical side--------------------

            if((((((float)pow(diffX, 2) / (float)pow(OZ3ellipse, 2)) + ((float)pow(diffY, 2) / (float)pow(tillOZ3, 2))) <= 1) && (diffX > 0)) &&
               ((((float)pow(diffX, 2) / (float)pow(OZ2ellipse, 2)) + ((float)pow(diffY, 2) / (float)pow(tillOZ2, 2))) > 1)){

                angle = atan2(diffY, (j - fovx));
                angleDeg = angle * (180 / M_PI) > 0 ? angle * (180 / M_PI) : 360 - abs(angle * (180 / M_PI));
                ElXLs = ((-1 * tillOZ2 * OZ2ellipse) / (sqrt(pow(tillOZ2, 2) + (pow(OZ2ellipse, 2) * pow(tan(angle), 2)))));
                ElYLs = (((angleDeg <= 180 ? 1 : -1) * tillOZ2 * OZ2ellipse) / (sqrt(pow(OZ2ellipse, 2) + (pow(tillOZ2, 2) / pow(tan(angle), 2)))));
                ElXGr = ((-1 * tillOZ3 * OZ3ellipse) / (sqrt(pow(tillOZ3, 2) + (pow(OZ3ellipse, 2) * pow(tan(angle), 2)))));
                ElYGr = (((angleDeg <= 180 ? 1 : -1) * tillOZ3 * OZ3ellipse) / (sqrt(pow(OZ3ellipse, 2) + (pow(tillOZ3, 2) / pow(tan(angle), 2)))));
                SOLGr = pow(ElXGr, 2) + pow(ElYGr, 2);
                SOLLs = pow(ElXLs, 2) + pow(ElYLs, 2);
                // First, we find out just how far between the two bands the current radius is
                radProg = ((double)((SOL) - SOLLs) / (double)(SOLGr - SOLLs));



                // Then we calculate the length of the subdivisions within the band
                divLength_df = 1 / (float)((rgcparams.divFactors[5] - rgcparams.divFactors[4]));
                divLength_mrfCen = 1 / (float)((oz3CenSide_m - oz2CenSide_m) > 0 ? (oz3CenSide_m - oz2CenSide_m) : -1);
                divLength_mrfSurr = 1 / (float)((oz3SurrSide_m - oz2SurrSide_m) > 0 ? (oz3SurrSide_m - oz2SurrSide_m) : -1);
                divLength_prfCen = 1 / (float)((oz3CenSide_p - oz2CenSide_p) > 0 ? (oz3CenSide_p - oz2CenSide_p) : -1);
                divLength_prfSurr = 1 / (float)((oz3SurrSide_p - oz2SurrSide_p) > 0 ? (oz3SurrSide_p - oz2SurrSide_p) : -1);

                // Calculation of respective RF sectors
                divSector_mrfCen = (int)(radProg / divLength_mrfCen);
                divSector_mrfSurr = (int)(radProg / divLength_mrfSurr);
                divSector_prfCen = (int)(radProg / divLength_prfCen);
                divSector_prfSurr = (int)(radProg / divLength_prfSurr);

                currRand = uni(rng);
                // ------------ Midget Centre ------------
                if(divLength_mrfCen == -1){
                    tmpCenSide_m = oz3CenSide_m;
                }else {
                    if(radProg > 1 - divLength_mrfCen){
                        if (currRand < ((radProg - (1 - divLength_mrfCen)) / divLength_mrfCen)) {
                            tmpCenSide_m = oz3CenSide_m;
                        }
                        else {
                            tmpCenSide_m = oz3CenSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfCen * divLength_mrfCen)) / divLength_mrfCen)){
                            tmpCenSide_m = oz2CenSide_m + divSector_mrfCen;
                        } else {
                            tmpCenSide_m = oz2CenSide_m + divSector_mrfCen + 1;
                        }
                    }
                }

                // ------------ Midget Surround------------
                if(divLength_mrfSurr == -1){
                    tmpSurrSide_m = oz3SurrSide_m;
                } else {
                    if(radProg > 1 - divLength_mrfSurr){
                        if (currRand < ((radProg - (1 - divLength_mrfSurr)) / divLength_mrfSurr)) {
                            tmpSurrSide_m = oz3SurrSide_m;
                        }
                        else {
                            tmpSurrSide_m = oz3SurrSide_m - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_mrfSurr * divLength_mrfSurr)) / divLength_mrfSurr)){
                            tmpSurrSide_m = oz2SurrSide_m + divSector_mrfSurr;
                        } else {
                            tmpSurrSide_m = oz2SurrSide_m + divSector_mrfSurr + 1;
                        }
                    }
                }

                // ------------ Parasol Centre ------------
                if(divLength_prfCen == -1){
                    tmpCenSide_p = oz3CenSide_p;
                } else {
                    if(radProg > 1 - divLength_prfCen){
                        if (currRand < ((radProg - (1 - divLength_prfCen)) / divLength_prfCen)) {
                            tmpCenSide_p = oz3CenSide_p;
                        }
                        else {
                            tmpCenSide_p = oz3CenSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfCen * divLength_prfCen)) / divLength_prfCen)){
                            tmpCenSide_p = oz2CenSide_p + divSector_prfCen;
                        } else {
                            tmpCenSide_p = oz2CenSide_p + divSector_prfCen + 1;
                        }
                    }
                }

                // ------------ Parasol Surround ------------
                if(divLength_prfSurr == -1){
                    tmpSurrSide_p = oz3SurrSide_p;
                } else {
                    if(radProg > 1 - divLength_prfSurr){
                        if (currRand < ((radProg - (1 - divLength_prfSurr)) / divLength_prfSurr)) {
                            tmpSurrSide_p = oz3SurrSide_p;
                        }
                        else {
                            tmpSurrSide_p = oz3SurrSide_p - 1;
                        }
                    } else {
                        if (currRand > ((radProg - ((float)divSector_prfSurr * divLength_prfSurr)) / divLength_prfSurr)){
                            tmpSurrSide_p = oz2SurrSide_p + divSector_prfSurr;
                        } else {
                            tmpSurrSide_p = oz2SurrSide_p + divSector_prfSurr + 1;
                        }
                    }
                }

                // If it's the last band
                if(radProg > 1 - divLength_df) {
                    divSector_df = (int)(radProg / divLength_df);
                    if (currRand < ((radProg - (1 - divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[5];
                    }
                    else {
                        divR = rgcparams.divFactors[5] - 1;
                    }
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector_df = (int)(radProg / divLength_df);


                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector_df * divLength_df)) / divLength_df)) {
                        divR = rgcparams.divFactors[4] + divSector_df;
                    }

                        // if currRand is greater
                    else {
                        divR = rgcparams.divFactors[4] + divSector_df + 1;
                    }
                }
                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets_l[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                        tmpRGCdets_r[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.6){
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_l[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_l[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_l[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_l[rgcArrayHeight][tmpxSum].zone = OZ3;

                    // Right eye
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.6){
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = MOTION;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_p;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_p;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].cenRfSide = tmpCenSide_m;
                        tmpRGCdets_r[rgcArrayHeight][tmpxSum].surRfWidth = tmpSurrSide_m;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets_r[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets_r[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets_r[rgcArrayHeight][tmpxSum].zone = OZ3;
                    tmpxSum+=1;
                }
            }
            // ----------------------- X --------------------------
        }


        if(tmpxSum != 0){
            xWidthsHost[rgcArrayHeight] = tmpxSum;
            RGCcount += tmpxSum;
            for(int p = 0; p < (tmpxSum % 2 == 0 ? (tmpxSum / 2) : ((tmpxSum + 1) / 2)); p+= 1){
                tmpRGCdets_r[rgcArrayHeight][p].perspX = (perspWidth - 1) - tmpRGCdets_r[rgcArrayHeight][p].perspX;
                if(!(tmpxSum % 2 != 0 && p == ((tmpxSum + 1) / 2) - 1)) tmpRGCdets_r[rgcArrayHeight][(tmpxSum - 1) - p].perspX = (perspWidth - 1) - tmpRGCdets_r[rgcArrayHeight][(tmpxSum - 1) - p].perspX;
                swapVar = tmpRGCdets_r[rgcArrayHeight][p];
                tmpRGCdets_r[rgcArrayHeight][p] = tmpRGCdets_r[rgcArrayHeight][(tmpxSum - 1) - p];
                tmpRGCdets_r[rgcArrayHeight][(xWidthsHost[rgcArrayHeight] - 1) - p] = swapVar;
            }
            for(int p = 0; p < tmpxSum; p+= 1){
                tmpRGCdets_r[rgcArrayHeight][p].perspID = ((tmpRGCdets_r[rgcArrayHeight][p].perspY * perspWidth * 4) + (tmpRGCdets_r[rgcArrayHeight][p].perspX * 4));
            }

        }
        yMatchHost[i] = rgcArrayHeight;
    }
    rgcArrayHeight += 1;

    cout << "Total RGC count :" << RGCcount << endl;

    v2rp->xWid = xWidthsHost;
    v2rp->yMat = yMatchHost;
    *v2rp->rgcArrH = rgcArrayHeight;
    *v2rp->RGCcnt = RGCcount;
    *v2rp->perspH = perspHeight;

    //------- Prep - the 2D arrays needed to capture RGC input ----------------
    RGC** RGCdets_l = (RGC**) malloc(rgcArrayHeight * sizeof(RGC*)), ** RGCdetsPin = (RGC**) malloc(rgcArrayHeight * sizeof(RGC*)), **RGCDetsDev_l, **detsArray_h, **detsArray_d,
            **RGCdets_r = (RGC**) malloc(rgcArrayHeight * sizeof(RGC*)), **RGCDetsDev_r, **RGCDetsTemp_l = (RGC**) malloc(rgcArrayHeight * sizeof(RGC*)), **RGCDetsTemp_r = (RGC**) malloc(rgcArrayHeight * sizeof(RGC*));
    float **rgcInputsLeft_h, **rgcInputsRight_h, **rgcInputsLeft_d, **rgcInputsRight_d, **rgcInputPin_l, **rgcInputPin_r;
    detsArray_h = (RGC**)malloc(rgcArrayHeight * sizeof(RGC*));
    rgcInputsLeft_h = (float**)malloc(rgcArrayHeight * sizeof(float*));
    rgcInputsRight_h = (float**)malloc(rgcArrayHeight * sizeof(float*));
    rgcInputPin_l = (float**)malloc(rgcArrayHeight * sizeof(float*));
    rgcInputPin_r = (float**)malloc(rgcArrayHeight * sizeof(float*));
    cudaMalloc(&RGCDetsDev_l, rgcArrayHeight * sizeof(RGC*));
    cudaMalloc(&RGCDetsDev_r, rgcArrayHeight * sizeof(RGC*));
    cudaMalloc(&rgcInputsLeft_d, rgcArrayHeight * sizeof(float*));
    cudaMalloc(&rgcInputsRight_d, rgcArrayHeight * sizeof(float*));
    cudaMalloc(&xWidthsDev, perspHeight * sizeof(int));
    cudaMalloc(&yMatchDev, perspHeight * sizeof(int));

    // RGC tempRGC;
    for(int i = 0; i < rgcArrayHeight; i+=1){

        int *dCurrWeights, *fFutWeights, *dPrevVals;
        cudaMalloc((void**) &RGCdets_l[i], ((xWidthsHost[i] * sizeof(RGC))));
        cudaMalloc((void**) &RGCdets_r[i], ((xWidthsHost[i] * sizeof(RGC))));

        // Transfers parasol data
        for(int j = 0; j < xWidthsHost[i]; j += 1){
            if(tmpRGCdets_l[i][j].type == PARASOL){
                cudaMalloc((void**)&tmpRGCdets_l[i][j].parasolWeights_curr, ((int)pow(((2 * tmpRGCdets_l[i][j].surRfWidth) + tmpRGCdets_l[i][j].cenRfSide), 2) * sizeof (int)));
                cudaMalloc((void**)&tmpRGCdets_l[i][j].previousValues, ((int)pow(((2 * tmpRGCdets_l[i][j].surRfWidth) + tmpRGCdets_l[i][j].cenRfSide), 2) * sizeof (int)));
            }
            if(tmpRGCdets_r[i][j].type == PARASOL){
                cudaMalloc((void**)&tmpRGCdets_r[i][j].parasolWeights_curr, ((int)pow(((2 * tmpRGCdets_r[i][j].surRfWidth) + tmpRGCdets_r[i][j].cenRfSide), 2) * sizeof (int)));
                cudaMalloc((void**)&tmpRGCdets_r[i][j].previousValues, ((int)pow(((2 * tmpRGCdets_r[i][j].surRfWidth) + tmpRGCdets_r[i][j].cenRfSide), 2) * sizeof (int)));
            }
        }
        cudaMemcpy(RGCdets_l[i], tmpRGCdets_l[i], xWidthsHost[i] * sizeof(RGC), cudaMemcpyHostToDevice);
        cudaMemcpy(RGCdets_r[i], tmpRGCdets_r[i], xWidthsHost[i] * sizeof(RGC), cudaMemcpyHostToDevice);
        RGCdetsPin[i] = (RGC*) malloc(xWidthsHost[i] * sizeof(RGC));
        rgcInputPin_l[i] = (float*)malloc(xWidthsHost[i] * sizeof(float));
        cudaMalloc((void **)&rgcInputsLeft_h[i], xWidthsHost[i] * sizeof(float));
        cudaMalloc((void **)&rgcInputsRight_h[i], xWidthsHost[i] * sizeof(float));
        if(i % (rgcArrayHeight / 10) == 0){
            cout << (i / (rgcArrayHeight / 10)) * 10  << "%... ";
        }
    }
    cout << endl;

    cudaMemcpy(RGCDetsDev_l, RGCdets_l, rgcArrayHeight * sizeof(RGC*), cudaMemcpyHostToDevice);
    cudaMemcpy(RGCDetsDev_r, RGCdets_r, rgcArrayHeight * sizeof(RGC*), cudaMemcpyHostToDevice);
    cudaMemcpy(rgcInputsLeft_d, rgcInputsLeft_h, rgcArrayHeight * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(rgcInputsRight_d, rgcInputsRight_h, rgcArrayHeight * sizeof(float*), cudaMemcpyHostToDevice);

//    // -------------------------------------------------------------------
//
//
    // ---------------- Init - Frame stuff for world data capture -----------------------
    uint8_t* frame_data_left, *frame_data_right;
    // Colour printing
    //printf("\x1B[34m                         \tWidth : \033[0m"); cout << frame_width << endl;
    if (posix_memalign((void**)&frame_data_left, ALIGNMENT, frame_width * frame_height * 4) != 0) {
        cout << "ERROR!!" << endl;
        printf("Couldn't allocate frameLeft buffer\n");
    }
    if (posix_memalign((void**)&frame_data_right, ALIGNMENT, frame_width * frame_height * 4) != 0) {
        cout << "ERROR!!" << endl;
        printf("Couldn't allocate frameRight buffer\n");
    }
    // Read a new frameLeft and load it into texture
    int64_t pts;
    int u = 0, p_y = 0, corrID = 0;
    AVFrame* frameLeft = video_reader_read_frame(&vr_stateLeft, frame_data_left, &pts);
    AVFrame* frameRight = video_reader_read_frame(&vr_stateRight, frame_data_right, &pts);
    int height = frameLeft->height, width = frameLeft->width;
    uint8_t* dataLeft = new uint8_t [height * width * 4];
    uint8_t* dataRight = new uint8_t [height * width * 4];
    // -------------------------------------------------------------------

    // -------------------- Init - perspective parameter initialization -----------------------
    params.perspWidth = perspWidth;
    params.perspHeight = perspHeight;
    params.worldWidth = frameLeft->width;
    params.worldHeight = frameLeft->height;
    params.hoffset = 0;            // Horizontal offaxis amount as percentage for shift lens
    params.voffset = 0;
    params.antialias = 2;          // Supersampling antialiasing;
    params.antialias2 = 4;
    params.latmin = -M_PI/2;       // Support for a inset of an equirectangular
    params.latmax = M_PI/2;
    params.longmin = -M_PI;
    params.longmax = M_PI;
    params.perspfov = 120;
    params.transform = NULL;
    params.ntransform = 0;
    params.debug = false;

    // --------------------------------------------------------

    //----------------- Sample transformation --------------------------
    params.transform = static_cast<TRANSFORM *>(realloc(params.transform,
                                                        (params.ntransform + 1) * sizeof(TRANSFORM)));
    params.transform[params.ntransform].axis = ZPAN;
    params.transform[params.ntransform].value = (M_PI / 180)*(30);
    params.ntransform++;
    for (int j=0;j<params.ntransform;j++) {
        params.transform[j].cvalue = cos(params.transform[j].value);
        params.transform[j].svalue = sin(params.transform[j].value);
    }
    //--------------------------------------------------------





    cudaDeviceProp a{};
    cudaSetDevice(0);
    cudaGetDeviceProperties(&a, 0);
    std::cout << "Device Overlap : " << a.deviceOverlap << endl;

    // ------------- Init - Perspective and World frame variables in world2persp ------------------
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(0);


    ::uint8_t *worldLeft, *worldRight;
    ::uint8_t *currentFrameLeft, *currentFrameRight;
    ::uint8_t *ffmpegLY, *ffmpegLU, *ffmpegLV, *ffmpegRY, *ffmpegRU, *ffmpegRV;
    uint8_t *perspHost, *perspLeft, *perspRight, *persp1, *perspTest;
    TRANSFORM *devTrans;
    // CUDA variables for device 0
    cudaStream_t memStream1, memStream2, funcStream1, funcStream2 ;
    // ------------------------------------------------------------------------------

    // ------------------- Frustum calculation for world2persp ---------------------------------

    CalcFrustum();


    // ------------------------------------------------------------------------------------

    // -------------------- Prep - stuff for world2persp ----------------------------------
    cudaSetDevice(0);
    cudaStreamCreate ( &memStream1) ;
    cudaStreamCreate ( &funcStream1) ;
    cudaMalloc(&ffmpegLY, numOfPixels * sizeof(::uint8_t));
    cudaMalloc(&ffmpegRY, numOfPixels * sizeof(::uint8_t));
    cudaMalloc(&ffmpegLU, (numOfPixels / 4) * sizeof(::uint8_t));
    cudaMalloc(&ffmpegRU, (numOfPixels / 4) * sizeof(::uint8_t));
    cudaMalloc(&ffmpegLV, (numOfPixels / 4) * sizeof(::uint8_t));
    cudaMalloc(&ffmpegRV, (numOfPixels / 4) * sizeof(::uint8_t));
    cudaMalloc(&worldLeft, numOfPixels * 4 * sizeof(::uint8_t));
    cudaMalloc(&worldRight, numOfPixels * 4 * sizeof(::uint8_t));
    cudaMalloc(&perspLeft, (perspHeight * perspWidth) * 4 * sizeof(uint8_t));
    cudaMalloc(&perspRight, (perspHeight * perspWidth) * 4 * sizeof(uint8_t));
    cudaMalloc(&perspTest, (perspHeight * perspWidth) * 4 * sizeof(uint8_t));
    cudaMalloc(&persp1, (perspHeight * perspWidth) * 4 * sizeof(uint8_t));
    cudaMalloc(&devTrans, params.ntransform * sizeof(TRANSFORM));
    cudaMalloc(&retinaDivs, 6 * sizeof(int));
    cudaSetDevice(1);
    cudaStreamCreate(&memStream2);
    cudaStreamCreate ( &funcStream2);

    cudaSetDevice(0);


    currentFrameLeft = (uint8_t*)malloc(numOfPixels * 4 * sizeof(uint8_t));
    currentFrameRight = (uint8_t*)malloc(numOfPixels * 4 * sizeof(uint8_t));
    perspHost = (uint8_t*)malloc((perspHeight * perspWidth * 4) * sizeof(uint8_t));



    const unsigned int bytes = frame_height * frame_width * sizeof(uint8_t);
    cudaMallocHost((void**)&currentFrameLeft, bytes * 4);
    cudaMallocHost((void**)&currentFrameRight, bytes * 4);
    cudaMallocHost((void**)&perspHost, (perspHeight * perspWidth) * 4 * sizeof(uint8_t));
    // -------------------------------------------------------------------------------------

    // ----------------------------- Loading data into frames for use as World -----------------------


    vector<::uint8_t *> frameArrayLeft, frameArrayRight;
    int frameIndex = 0; int toIgnore = 0;


    cudaSetDevice(0);
    TRANSFORM *saccadeBank = NULL;
    int saccadeCount = 0;

    saccadeBank = static_cast<TRANSFORM *>(realloc(saccadeBank,(saccadeCount + 1) * sizeof(TRANSFORM)));
    saccadeBank[saccadeCount].axis = ZPAN;
    saccadeBank[saccadeCount].value = (M_PI / 180)*(20);
    saccadeCount+=1;
    for (int j=0;j<saccadeCount;j++) {
        saccadeBank[j].cvalue = cos(saccadeBank[j].value);
        saccadeBank[j].svalue = sin(saccadeBank[j].value);
    }

    params.transform = saccadeBank;
    params.ntransform = saccadeCount;


    cudaMalloc(&devTrans, saccadeCount * sizeof(TRANSFORM));


    while(true){


        cudaSetDevice(0);

        saccadeBank = static_cast<TRANSFORM *>(realloc(saccadeBank,(saccadeCount + 1) * sizeof(TRANSFORM)));
        saccadeBank[saccadeCount].axis = ZPAN;
        saccadeBank[saccadeCount].value = (M_PI / 180)*(-1);
        saccadeCount+=1;
        for (int j=0;j<saccadeCount;j++) {
            saccadeBank[j].cvalue = cos(saccadeBank[j].value);
            saccadeBank[j].svalue = sin(saccadeBank[j].value);
        }

        params.transform = saccadeBank;
        params.ntransform = saccadeCount;


        cudaMalloc(&devTrans, saccadeCount * sizeof(TRANSFORM));


        frameLeft = video_reader_read_frame(&vr_stateLeft, frame_data_left, &pts);
        frameRight = video_reader_read_frame(&vr_stateRight, frame_data_right, &pts);
        cudaMemcpy(ffmpegLY, frameLeft->data[0], numOfPixels * sizeof(::uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ffmpegRY, frameRight->data[0], numOfPixels * sizeof(::uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ffmpegLU, frameLeft->data[1], (numOfPixels / 4) * sizeof(::uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ffmpegRU, frameRight->data[1], (numOfPixels / 4) * sizeof(::uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ffmpegLV, frameLeft->data[2], (numOfPixels / 4) * sizeof(::uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ffmpegRV, frameRight->data[2], (numOfPixels / 4) * sizeof(::uint8_t), cudaMemcpyHostToDevice);
        cudaSetDevice(0);

//        params.transform = static_cast<TRANSFORM *>(realloc(params.transform,
//                                                            (params.ntransform + 1) * sizeof(TRANSFORM)));

        //cudaMalloc(&devTrans, params.ntransform * sizeof(TRANSFORM));




        auto start = std::chrono::high_resolution_clock::now();

        // --------------- Initial prep ----------------------
        cudaSetDevice(0);
        cudaMemcpy(devTrans, params.transform, params.ntransform * sizeof(TRANSFORM), cudaMemcpyHostToDevice);
        cudaMemcpy(retinaDivs, rgcparams.divFactors, 6 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(yMatchDev, yMatchHost, perspHeight * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(xWidthsDev, xWidthsHost, perspHeight * sizeof(int), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        rgcparams.divFactors = retinaDivs;
        rgcparams.rgcArrLen = rgcArrayHeight;
        params.vLineSize = frameLeft->linesize[2];
        params.transform = devTrans;


        if(inputMode == VID){
            // Changes the ffmpeg YUV frames to RGB WorldFrames
            ffmpeg2Persp <<<(perspNumofPixels+ 1023)/1024, 1024, 0, funcStream1>>> (perspLeft, perspRight, ffmpegLY, ffmpegRY,
                                                                                    ffmpegLU, ffmpegRU, ffmpegLV, ffmpegRV, params);
            cudaDeviceSynchronize();

        } else if (inputMode == VR) {
            // Changes the ffmpeg YUV frames to RGB WorldFrames
            ffmpeg2World <<<(numOfPixels + 1023)/1024, 1024, 0, funcStream1>>> (worldLeft, worldRight, ffmpegLY, ffmpegRY,
                                                                                ffmpegLU, ffmpegRU, ffmpegLV, ffmpegRV, params);
            cudaDeviceSynchronize();

            // Converts vr 360 video into a perspective frame of dimensions perspHeight x perspWidth
            world2Persp<<<((perspHeight * perspWidth) + 1023)/1024, 1024, 0, funcStream1>>>(worldLeft, perspLeft, worldRight, perspRight, params, frustum);
            cudaDeviceSynchronize();
        }


        // Creates test frame
        createPerspTest<<<((perspHeight * perspWidth) + 1023)/1024, 1024, 0, funcStream1>>>(perspTest, perspLeft, params);
        cudaDeviceSynchronize();


        // Forms RGC inputs
        formRGCcurrents_l<<<(RGCcount + 1023) / 1024, 1024, 0, funcStream1>>>(rgcparams, perspTest, perspLeft,
                                                                              rgcInputsLeft_d, xWidthsDev, yMatchDev,
                                                                              RGCDetsDev_l, toIgnore);
        cudaDeviceSynchronize();

//        // Forms RGC inputs
        formRGCcurrents_r<<<(RGCcount + 1023) / 1024, 1024, 0, funcStream1>>>(rgcparams, perspTest, perspRight,
                                                                              rgcInputsRight_d, xWidthsDev, yMatchDev,
                                                                              RGCDetsDev_r, toIgnore);
        cudaDeviceSynchronize();

        cudaSetDevice(0);
        cudaDeviceSynchronize();
        cudaSetDevice(1);
        cudaDeviceSynchronize();


        // Transfers computed values back to host
        cudaSetDevice(0);

        // Transfer perspective frame back after computation - not necessary
        cudaMemcpy(perspHost, perspTest, (perspHeight * perspWidth * 4 ) * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        //else cudaMemcpy(perspHost, perspTest, (perspHeight * perspWidth * 4 ) * sizeof(uint8_t), cudaMemcpyDeviceToHost);

//
//
//
        // Transfers RGC details array back after initialisation
        if(toIgnore == 0){
            cudaMemcpy(RGCdets_l, RGCDetsDev_l, rgcArrayHeight * sizeof(RGC*), cudaMemcpyDeviceToHost);
            for(int p = 0; p < rgcArrayHeight; p+=1){
                cudaMemcpy(RGCdetsPin[p], RGCdets_l[p], xWidthsHost[p] * sizeof(RGC), cudaMemcpyDeviceToHost);
            }
            rgcDetsQ->push(RGCdetsPin);
            cudaMemcpy(RGCdets_r, RGCDetsDev_r, rgcArrayHeight * sizeof(RGC*), cudaMemcpyDeviceToHost);
            for(int p = 0; p < rgcArrayHeight; p+=1){
                cudaMemcpy(RGCdetsPin[p], RGCdets_r[p], xWidthsHost[p] * sizeof(RGC), cudaMemcpyDeviceToHost);
            }
            rgcDetsQ->push(RGCdetsPin);
        }
//        for(int b = 0; b < 100; b+=1){
//            for(int a = 0; a < xWidthsHost[b]; a+=1){
//                cout << RGCdetsPin[b][a].perspX << " - ";
//            }
//        }



        rgcInputPin_l = (float**)malloc(rgcArrayHeight * sizeof(float*));
        rgcInputPin_r = (float**)malloc(rgcArrayHeight * sizeof(float*));
        for(int i = 0; i < rgcArrayHeight; i+=1){
            rgcInputPin_l[i] = (float*)malloc(xWidthsHost[i] * sizeof(float));
            rgcInputPin_r[i] = (float*)malloc(xWidthsHost[i] * sizeof(float));
        }

        // Transfers RGC inputs back - required if doing neural computation on another GPU
        cudaMemcpy(rgcInputsLeft_h, rgcInputsLeft_d, rgcArrayHeight * sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(rgcInputsRight_h, rgcInputsRight_d, rgcArrayHeight * sizeof(float*), cudaMemcpyDeviceToHost);
        for(int p = 0; p < 800; p+=1){
            cudaMemcpy(rgcInputPin_l[p], rgcInputsLeft_h[p], xWidthsHost[p] * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(rgcInputPin_r[p], rgcInputsRight_h[p], xWidthsHost[p] * sizeof(float), cudaMemcpyDeviceToHost);

//        for(int j = 0; j < xWidthsHost[p]; j+=1){
//            if(rgcInputPin_l[p][j] != j){
//                cout << "Error at : " << p << " Index of Error is : " << j  << " val : " << rgcInputPin_l[p][j] << endl;
//            }
//        }
// ---------------------------------- Print RGC vals ---------------------------------
//            cout << "i : " << p << " || Length : " << xWidthsHost[p] << " || ";
//            for(int j = 0; j < xWidthsHost[p]; j+=1){
//                if(true){
//                    if(RGCdetsPin[p][j].detType == LUM && RGCdetsPin[p][j].type == MIDGET){
//                        printf("\033[1;31m%f\033[0m", rgcInputPin_l[p][j]);
//                        cout << " - ";
//                    } else {
//                        cout << rgcInputPin_l[p][j] << " - ";
//                    }
//                }
//            }
//            cout << endl;
// ------------------------------------------------------------------------------------
        }
        {
            lock_guard<mutex> lock(rgcMut);
            rgcQueue_l->push(rgcInputPin_l);
            rgcQueue_r->push(rgcInputPin_r);
            *frameNum = *frameNum + 1;
        }
        // cout << "Current Size : " << rgcQueue_l->size() << std::endl;
        rgcCond.notify_all();





        glBindTexture(GL_TEXTURE_2D, tex_handle);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, perspWidth, perspHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, perspHost);

        // Render whatever you want
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex_handle);
        glBegin(GL_QUADS);
        glTexCoord2d(0,0); glVertex2i(0, 0);
        glTexCoord2d(1,0); glVertex2i(0 + camWidth, 0);
        glTexCoord2d(1,1); glVertex2i(0 + camWidth, 0 + camHeight);
        glTexCoord2d(0,1); glVertex2i(0, 0 + camHeight);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        glfwSwapBuffers(window);
        glfwPollEvents();
        // ::getchar();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<chrono::microseconds>(stop - start).count();
        cout << "Net duration of visual pass : " << duration << endl;
        toIgnore+=1;
//        cudaFree(devTrans);
//        cudaFree(retinaDivs);
//        cudaFree(yMatchDev);
//        cudaFree(xWidthsDev);
    }


    // ---------------------------------------------------------------------------------










    // cout << "RGC count : " << RGCcount << endl;



    ;
    // cudaFree(Y_1);
//    cudaFree(Y_1);
//    cudaFree(Y_2);


}

// world2PerspTest<<<((perspHeight * perspWidth * 4) + 1023)/1024, 1024, 0, funcStream1>>>(perspLeft, persp1);
//    for(int i = 0; i < 1000; i+=1){
//
//        if(i%36 == 1){
////            frameLeft = video_reader_read_frame(&vr_stateLeft, frame_data_left, &pts);
////            frameArrayLeft.push_back(frameLeft->dataLeft[0]);
//            ::memcpy(currentFrameLeft, frameArrayLeft[frameStorageCountr], numOfPixels*sizeof(::uint8_t));
//            // Transfers dataLeft from Video array to local pinned memory
//            // cudaMemcpyAsync(currentFrameLeft, frameArrayLeft[0], numOfPixels * sizeof(::uint8_t), cudaMemcpyHostToHost, memStream1);
//            cudaSetDevice(0);
//            cudaMemcpyAsync(worldLeft, currentFrameLeft, numOfPixels * 4 *sizeof(::uint8_t), cudaMemcpyHostToDevice, memStream1);
////            cudaMemcpyAsync(U_1, currentFrameLeft, numOfPixels*sizeof(::uint8_t), cudaMemcpyHostToDevice, memStream1);
////            cudaMemcpyAsync(V_1, currentFrameLeft, numOfPixels*sizeof(::uint8_t), cudaMemcpyHostToDevice, memStream1);
//            frameStorageCountr++;
////            cudaSetDevice(0);
////            cudaMemcpyAsync(currentFrameLeft, frameArrayLeft[0], numOfPixels * sizeof(::uint8_t), cudaMemcpyHostToHost, memStream1);
//        }
//        if(i%36 == 35){
//            cudaSetDevice(0);
//            cudaDeviceSynchronize();

//        cout << "i : " << p << " || Length : " << xWidthsHost[p] << " || ";
//        for(int j = 0; j < xWidthsHost[p]; j+=1){
//            if(RGCdetsPin[p][j].detType == LUM) cout << ".";
//            else {
//                if (RGCdetsPin[p][j].colID == R_rgc) printf("\033[1;31m.\033[0m");
//                if (RGCdetsPin[p][j].colID == G_rgc) printf("\033[1;32m.\033[0m");
//                if (RGCdetsPin[p][j].colID == B_rgc) printf("\033[1;34m.\033[0m");
//                if (RGCdetsPin[p][j].colID == Y_rgc) printf("\033[1;33m.\033[0m");
//            }
//
//            cout << " ";
//        }
//        cout << endl;

// To debug RGCdets creation
//    if(j < fovx && i < fovy && fovy - i < 30) cout << "x : " << j - fovx << " || y :" << diffY << " || angle : " << angleDeg << " || divL : " <<
//    divLength_df <<  " || sector_df : " << divSector_df <<  " || sector_mrfCen : " << divSector_mrfCen << " || divLen_mrfCen : "
//    << divLength_mrfCen << " || divLen_mrfSurr : " << divLength_mrfSurr << " || sector_mrfSurr : " << divSector_mrfSurr
//    <<" || CenSide : " << tmpCenSide_m << " || SurrSide : " << tmpSurrSide_m << " || radProg : " << radProg << " || divR : " << divR << endl;
