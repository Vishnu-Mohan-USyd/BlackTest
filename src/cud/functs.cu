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

using namespace matplot;
using namespace std;

PARAMS params;
FRUSTUM frustum;
RGCPARAMS rgcparams;
RGCdev rgcdev;

__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (n -1)) x[i] = x[i+1] ;


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
void world2PerspTest(::uint8_t  *perspFrame, ::uint8_t  *persp1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    persp1[i] = perspFrame[i];

}



__global__
void formRGCinputs(int foveaPoint, int frameH, int frameW , RGCPARAMS rgcparams, uint8_t  *perspLeft, uint8_t  *perspRight, float** rgcLeftDevice, float** rgcRightDevice, int* xWidths, int* yMatch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Here we aim to divide i into different retinal compartments
     *
     * ---------------- For 8k Video --------------
     *
     * Fovea - 300 x 300 || Total RGCs are 90,000
     * Parafovea - 200 plus on all sides || Total RGCs are 100,150
     * Perifovea - 300 plus on all sides || Total RGCs are 133,200
     * Outer Zone 1 - 400 up, 500 side || Total RGCs are 196,250 (3.14M pixels)
     * Outer Zone 2 - 500 up, 1000 side || Total RGCs are 132,625 (8.25M pixels)
     * Outer Zone 3 - 600 up, 1500 side || Total RGCs are 125,300 (18.06M pixels)
     *
     * Total rgcLeft collection count - 777,526 RGCs
     *
     * ---------------- For 4k Video --------------
     *
     * Fovea - 180 x 180 || Total RGCs are 32,400
     * Parafovea - 100 plus on all sides || Total RGCs are 100,150
     * Perifovea - 300 plus on all sides || Total RGCs are 133,200
     * Outer Zone 1 - 400 up, 500 side || Total RGCs are 196,250 (3.14M pixels)
     * Outer Zone 2 - 500 up, 1000 side || Total RGCs are 132,625 (8.25M pixels)
     * Outer Zone 3 - 600 up, 1500 side || Total RGCs are 125,300 (18.06M pixels)
     *
     * Total rgcLeft collection count - 777,526 RGCs
     * */

    // foveaPoint = pixelCount/2;
    int defFovPoint = 4145280;
    int actFovY = foveaPoint/frameW;
    int actFovX = foveaPoint - (actFovY * frameW);
    int fovY = (rgcparams.perspHeight / 2) - 1;
    int fovX = (rgcparams.perspWidth / 2) - 1;
    int currY = i/frameW;
    int currX = i - (currY * frameW);
    // rgcLeft[i] = 2;
    int xDiff = currX - fovX;
    int yDiff = currY - fovY;
    int index = 0;
    int rgcX, rgcY;
    if((((xDiff < 0) && (((xDiff - 8) * -1) > actFovX)) || (((xDiff) > 0) && ((xDiff + 8) > (frameW - actFovX)))) ||
       ((((yDiff) < 0) && (((yDiff - 8) * -1) > actFovY)) || (((yDiff) > 0) && ((yDiff + 8) > (frameH - actFovY))))){
        index = -1;
    } else {
        index = (xDiff + actFovX) + ((yDiff + actFovY) * frameW);
    }


    int fovWidth = rgcparams.foveaWidth; int fovWidthMidPre = ((fovWidth / 2) - 1); int fovWidthMidPost = (fovWidth / 2);
    int paraLength = rgcparams.paraLength;
    int periLength = rgcparams.periLength;







    // ----------------------- Foveal Processing -----------------------
    if((currX >= fovX - fovWidthMidPre) && (currY >= fovY - fovWidthMidPre) && (currX <= fovX + fovWidthMidPost) && (currY <= fovY + fovWidthMidPost)){
        // rgcLeft[2] = 1;
        int RGCoffset = 0;
        int sectionWidth = fovWidth;
        float midLeft = ((float)perspLeft[i]) / 255;
        float surLeft = ((0.125 * (float)perspLeft[(i - frameW) - 1]) + (0.125 * (float)perspLeft[(i - frameW)]) + (0.125 * (float)perspLeft[(i - frameW) + 1]) +
                         (0.125 * (float)perspLeft[(i) - 1]) + (0.125 * (float)perspLeft[(i) + 1]) +
                         (0.125 * (float)perspLeft[(i + frameW) - 1]) + (0.125 * (float)perspLeft[(i + frameW)]) + (0.125 * (float)perspLeft[(i + frameW) + 1])) / 255;
        float res  = midLeft - surLeft;
        if(index < 0) res = -1;
        else res  = midLeft - surLeft;
        if(res < 0) res = 0;

        // Converting from 2D modelled frame to linear RGC array
        rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +
               (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * (rgcparams.oz2side / rgcparams.divFactors[4])) +
               (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * (rgcparams.oz1side / rgcparams.divFactors[3])) +
               (((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * (rgcparams.periLength / rgcparams.divFactors[2])) +
               (((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * (rgcparams.paraLength / rgcparams.divFactors[1])) +
               (currX - (fovX - (fovWidthMidPre)));

        rgcLeftDevice[yMatch[currY]][rgcX] = rgcX;
    }

    // ---------------------- Parafoveal Processing --------------------

    if (// Inner Perimeters

        // Left
            ((currX < (fovX - fovWidthMidPre)) && (currX >= (fovX - (fovWidthMidPre + paraLength))) && (currY >= (fovY - (fovWidthMidPre + paraLength))) && (currY <= (fovY + (fovWidthMidPost + paraLength)))) ||
            // Top
            ((currX >= (fovX - (fovWidthMidPre + paraLength))) && (currX <= (fovX + (fovWidthMidPost + paraLength))) && (currY >= (fovY - (fovWidthMidPre + paraLength))) && (currY < (fovY - fovWidthMidPre))) ||
            // Right
            ((currX > (fovX + fovWidthMidPost)) && (currX <= (fovX + (fovWidthMidPost + paraLength))) && (currY >= (fovY - (fovWidthMidPre + paraLength))) && (currY <= (fovY + (fovWidthMidPost + paraLength)))) ||
            // Bottom
            ((currX >= (fovX - (fovWidthMidPre + paraLength))) && (currX <= (fovX + (fovWidthMidPost + paraLength))) && (currY > (fovY + fovWidthMidPost)) && (currY <= (fovY + (fovWidthMidPost + paraLength))))){

        //rgcLeft[90001] = 2;
        int RGCoffset = fovWidth * fovWidth;
        int sectionWidth = 0;
        int currentSum = 0;
        int checkr = 0;
//        if (((currY >= (fovY - (149 + 200))) && (currY <= (fovY - 149))) ||
//                ((currX >= (fovX - (149 + 200))) && (currX <= (fovX + (150 + 200))) && (currY >= (fovY + 150)) && (currY <= (fovY + (149 + 200)))))
//            sectionWidth = 350;
//        else if ((currY >= (fovY - (149))) && (currY <= (fovY + (149))))
//            sectionWidth = 201;

        float midLeft = ((float)perspLeft[i])/255;
        float surLeft = ((0.125 * (float)perspLeft[(i - frameW) - 1]) + (0.125 * (float)perspLeft[(i - frameW)]) + (0.125 * (float)perspLeft[(i - frameW) + 1]) +
                         (0.125 * (float)perspLeft[(i) - 1]) + (0.125 * (float)perspLeft[(i) + 1]) +
                         (0.125 * (float)perspLeft[(i + frameW) - 1]) + (0.125 * (float)perspLeft[(i + frameW)]) + (0.125 * (float)perspLeft[(i + frameW) + 1]))/255;
        float res  = midLeft - surLeft;
        if(index < 0) res = -1;
        else res  = midLeft - surLeft;

        // Picking out units of computation
        if (((currX % rgcparams.divFactors[1]) == 0) && ((currY % rgcparams.divFactors[1]) == 0)){

            // To check if the current index is in the middle,vertically
            if((currY >= (fovY - (fovWidthMidPre))) && (currY <= (fovY + (fovWidthMidPost)))){

                // -------------- Translating Indices ----------------
                //Left Section
                if(currX < fovX){
                    checkr = 1;
                    rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +
                           (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * (rgcparams.oz2side / rgcparams.divFactors[4])) +
                           (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * (rgcparams.oz1side / rgcparams.divFactors[3])) +
                           (((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * (rgcparams.periLength / rgcparams.divFactors[2])) +
                           (((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * ((currX - (fovX - (fovWidthMidPre + paraLength))) / rgcparams.divFactors[1]));
                }
                    // Right Section
                else {
                    checkr = 2;
                    rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (((currY > 0) && (currY < rgcparams.perspHeight)) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +

                           (int)((((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * (((currY >= ((double)rgcparams.oz3up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up)))) ? 1 : 0) * ((double)rgcparams.oz2side / (double)rgcparams.divFactors[4])) +
                                 (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength) + (2 * (double)rgcparams.oz1side)) / (double)rgcparams.divFactors[4]))) +


                           (int)((((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up)))) ? 1 : 0) * ((double)rgcparams.oz1side / (double)rgcparams.divFactors[3])) +
                                 (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength)) / (double)rgcparams.divFactors[3]))) +


                           (int)((((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)))) ? 1 : 0) * ((double)rgcparams.periLength / (double)rgcparams.divFactors[2])) +
                                 (((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength)) / (double)rgcparams.divFactors[2]))) +


                           (int)((((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)))) ? 1 : 0) * ((double)rgcparams.paraLength / (double)rgcparams.divFactors[1])) +
                                 (((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength + paraLength))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength + paraLength))))) ? 1 : 0) * (((double)rgcparams.foveaWidth) / (double)rgcparams.divFactors[1]))) +


                            (rgcparams.foveaWidth * (((currY >= (rgcparams.oz3up + rgcparams.oz2up + rgcparams.oz1up + periLength + paraLength)) && (currY < (rgcparams.perspHeight - (rgcparams.oz3up + rgcparams.oz2up + rgcparams.oz1up + periLength + paraLength)))) ? 1 : 0)) +
                           (((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * ((currX - (fovX + (fovWidthMidPost + 1))) / rgcparams.divFactors[1]));
                }
            }

                // Checks if current index is in the top or bottom sections
            else {
                rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +
                       (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * (rgcparams.oz2side / rgcparams.divFactors[4])) +
                       (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * (rgcparams.oz1side / rgcparams.divFactors[3])) +
                       (((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * (rgcparams.periLength / rgcparams.divFactors[2])) +
                       (((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * ((currX - (fovX - (fovWidthMidPre + paraLength))) / rgcparams.divFactors[1]));

            }

            // Entering data
            rgcLeftDevice[yMatch[currY]][rgcX] = rgcX;
        }
    }
    // ---------------------------------- X -----------------------------------


    //------------------------ Perifoveal Processing -------------------------
    if (// Inner Perimeters

        // Left
            ((currX < (fovX - (fovWidthMidPre + paraLength))) && (currX >= (fovX - (fovWidthMidPre + paraLength + periLength))) && (currY >= (fovY - (fovWidthMidPre + paraLength + periLength))) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength)))) ||
            // Top
            ((currX >= (fovX - (fovWidthMidPre + paraLength + periLength))) && (currX <= (fovX + (fovWidthMidPost + paraLength + periLength))) && (currY >= (fovY - (fovWidthMidPre + paraLength + periLength))) && (currY < (fovY - (fovWidthMidPre + paraLength)))) ||
            // Right
            ((currX > (fovX + fovWidthMidPost + paraLength)) && (currX <= (fovX + (fovWidthMidPost + paraLength + periLength))) && (currY >= (fovY - (fovWidthMidPre + paraLength + periLength))) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength)))) ||
            // Bottom
            ((currX >= (fovX - (fovWidthMidPre + paraLength + periLength))) && (currX <= (fovX + (fovWidthMidPost + paraLength + periLength))) && (currY > (fovY + fovWidthMidPost + paraLength)) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength))))){
        int RGCoffset = (fovWidth * fovWidth) + (((((fovWidth + (2 * paraLength)) / rgcparams.divFactors[1])) * ( paraLength / rgcparams.divFactors[1] )) + ((paraLength) * (fovWidth/rgcparams.divFactors[1])) + ((((fovWidth + (2 * paraLength)) / rgcparams.divFactors[1]) * (paraLength / rgcparams.divFactors[1]))));
        int sectionWidth = 0;
        int currentSum = 0;
        int checkr = 0;

        float midLeft = ((float)perspLeft[i])/255;
        float surLeft = (
                                // Center - 2
                                (0.0416667 * (float)perspLeft[(i - (2 * frameW)) - 2]) + (0.0416667 * (float)perspLeft[(i - (2 * frameW)) - 1]) + (0.0416667 * (float)perspLeft[(i - (2 * frameW))]) + (0.0416667 * (float)perspLeft[(i - (2 * frameW)) + 1]) + (0.0416667 * (float)perspLeft[(i - (2 * frameW)) + 2]) +
                                // Center - 1
                                (0.0416667 * (float)perspLeft[(i - frameW) - 2]) +(0.0416667 * (float)perspLeft[(i - frameW) - 1]) + (0.0416667 * (float)perspLeft[(i - frameW)]) + (0.0416667 * (float)perspLeft[(i - frameW) + 1]) + (0.0416667 * (float)perspLeft[(i - frameW) + 2]) +
                                // Center
                                (0.0416667 * (float)perspLeft[(i) - 2]) + (0.0416667 * (float)perspLeft[(i) - 1]) + (0.0416667 * (float)perspLeft[(i) + 1]) + (0.0416667 * (float)perspLeft[(i) + 2]) +
                                // Center + 1
                                (0.0416667 * (float)perspLeft[(i) + frameW - 2]) + (0.0416667 * (float)perspLeft[(i) + frameW - 1]) + (0.0416667 * (float)perspLeft[(i) + frameW]) +(0.0416667 * (float)perspLeft[(i) + frameW + 1]) + (0.0416667 * (float)perspLeft[(i) + frameW + 2]) +
                                // Center + 2
                                (0.0416667 * (float)perspLeft[(i + (2 * frameW)) - 2]) + (0.0416667 * (float)perspLeft[(i + (2 * frameW)) - 1]) + (0.0416667 * (float)perspLeft[(i + (2 * frameW))]) + (0.0416667 * (float)perspLeft[(i + (2 * frameW)) + 1]) + (0.0416667 * (float)perspLeft[(i + (2 * frameW)) + 2]))/255;
        float res  = midLeft - surLeft;
        if(index < 0) res = -1;
        else res  = midLeft - surLeft;


        int compIndex = 0;

        // Picking out units of computation
        if (((currX) % rgcparams.divFactors[2] == 0) && ((currY) % rgcparams.divFactors[2] == 0)){

            // To check if the current index is in the middle,vertically
            if((currY >= (fovY - (fovWidthMidPre + paraLength))) && (currY <= (fovY + (fovWidthMidPost + paraLength)))){

                // -------------- Translating Indices ----------------
                //Left Section
                if(currX < fovX){
                    checkr = 1;
                    rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +
                           (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * (rgcparams.oz2side / rgcparams.divFactors[4])) +
                           (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * (rgcparams.oz1side / rgcparams.divFactors[3])) +
                           (((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * ((currX - (fovX - (fovWidthMidPre + paraLength + periLength))) / rgcparams.divFactors[2]));
                }
                    // Right Section
                else {
                    checkr = 2;
                    rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (((currY > 0) && (currY < rgcparams.perspHeight)) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +

                            (int)((((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * (((currY >= ((double)rgcparams.oz3up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up)))) ? 1 : 0) * ((double)rgcparams.oz2side / (double)rgcparams.divFactors[4])) +
                                  (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength) + (2 * (double)rgcparams.oz1side)) / (double)rgcparams.divFactors[4]))) +


                            (int)((((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up)))) ? 1 : 0) * ((double)rgcparams.oz1side / (double)rgcparams.divFactors[3])) +
                                  (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength)) / (double)rgcparams.divFactors[3]))) +


                            (int)((((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)))) ? 1 : 0) * ((double)rgcparams.periLength / (double)rgcparams.divFactors[2])) +
                                  (((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength)) / (double)rgcparams.divFactors[2]))) +


                            (int)((((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * 2 * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)))) ? 1 : 0) * ((double)rgcparams.paraLength / (double)rgcparams.divFactors[1])) +
                                  (((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength + paraLength))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength + paraLength))))) ? 1 : 0) * (((double)rgcparams.foveaWidth) / (double)rgcparams.divFactors[1]))) +


                            (rgcparams.foveaWidth * (((currY >= (rgcparams.oz3up + rgcparams.oz2up + rgcparams.oz1up + periLength + paraLength)) && (currY < (rgcparams.perspHeight - (rgcparams.oz3up + rgcparams.oz2up + rgcparams.oz1up + periLength + paraLength)))) ? 1 : 0)) +
                           (((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * ((currX - (fovX + (fovWidthMidPost + paraLength + 1))) / rgcparams.divFactors[2]));
                }
            }

                // Checks if current index is in the top or bottom sections
            else {
                rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +
                       (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * (rgcparams.oz2side / rgcparams.divFactors[4])) +
                       (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * (rgcparams.oz1side / rgcparams.divFactors[3])) +
                       (((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * ((currX - (fovX - (fovWidthMidPre + paraLength + periLength))) / rgcparams.divFactors[2]));

            }

            // Entering data
            rgcLeftDevice[yMatch[currY]][rgcX] = rgcX;
        }
    }

    // ------------------------------------------ OZ1 processing -------------------------------------

    if (// Inner Perimeters

        // Left
            ((currX < (fovX - (fovWidthMidPre + paraLength + periLength))) && (currX >= (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side))) && (currY >= (fovY - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1up))) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1up)))) ||
            // Top
            ((currX >= (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side))) && (currX <= (fovX + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1side))) && (currY >= (fovY - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1up))) && (currY < (fovY - (fovWidthMidPre + paraLength  + periLength)))) ||
            // Right
            ((currX > (fovX + fovWidthMidPost + paraLength + periLength)) && (currX <= (fovX + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1side))) && (currY >= (fovY - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1up))) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1up)))) ||
            // Bottom
            ((currX >= (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side))) && (currX <= (fovX + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1side))) && (currY > (fovY + fovWidthMidPost + paraLength + periLength)) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1up))))){

        int RGCoffset = (fovWidth * fovWidth) + (((((fovWidth + (2 * paraLength)) / rgcparams.divFactors[1])) * ( paraLength / rgcparams.divFactors[1] )) + ((paraLength) * (fovWidth/rgcparams.divFactors[1])) + ((((fovWidth + (2 * paraLength)) / rgcparams.divFactors[1]) * (paraLength / rgcparams.divFactors[1])))) +
                        (((((fovWidth + (paraLength * 2) + (periLength * 2)) / rgcparams.divFactors[2]) * (periLength / rgcparams.divFactors[2])) + (((2 * periLength) / rgcparams.divFactors[2]) * ((fovWidth + (2 * paraLength)) / rgcparams.divFactors[2]))) + ((((fovWidth + (paraLength * 2) + (periLength * 2)) / rgcparams.divFactors[2])) * (periLength / rgcparams.divFactors[2])));
        int sectionWidth = 0;
        int currentSum = 0;
        int checkr = 0;

        float midLeft = (0.25 * (((float)perspLeft[i])/255)) + (0.25 * (((float)perspLeft[i + 1])/255)) + (0.25 * (((float)perspLeft[i + frameW])/255)) + (0.25 * (((float)perspLeft[i + frameW + 1])/255));
        float surLeft = ((0.03 * (float)perspLeft[(i - (2 * frameW)) - 2]) + (0.03 * (float)perspLeft[(i - (2 * frameW)) - 1]) + (0.03 * (float)perspLeft[(i - (2 * frameW))]) + (0.03 * (float)perspLeft[(i - (2 * frameW)) + 1]) + (0.03 * (float)perspLeft[(i - (2 * frameW)) + 2]) + (0.03 * (float)perspLeft[(i - (2 * frameW)) + 3]) +
                         (0.03 * (float)perspLeft[(i - frameW) - 2]) + (0.03 * (float)perspLeft[(i - frameW) - 1]) + (0.03 * (float)perspLeft[(i - frameW)]) + (0.03 * (float)perspLeft[(i - frameW) + 1]) + (0.03 * (float)perspLeft[(i - frameW) + 2]) + (0.03 * (float)perspLeft[(i - frameW) + 3]) +
                         (0.03 * (float)perspLeft[(i) - 2])  + (0.03 * (float)perspLeft[(i) - 1]) + (0.03 * (float)perspLeft[(i) + 2]) + (0.03 * (float)perspLeft[(i) + 3]) + (0.03 * (float)perspLeft[(i) + frameW - 2]) + (0.03 * (float)perspLeft[(i) + frameW - 1]) + (0.03 * (float)perspLeft[(i) + frameW + 2]) + (0.03 * (float)perspLeft[(i) + frameW + 3]) +
                         (0.03 * (float)perspLeft[(i + (2 * frameW)) - 2]) + (0.03 * (float)perspLeft[(i + (2 * frameW)) - 1]) + (0.03 * (float)perspLeft[(i + (2 * frameW))]) + (0.03 * (float)perspLeft[(i + (2 * frameW)) + 1]) + (0.03 * (float)perspLeft[(i + (2 * frameW)) + 2]) + (0.03 * (float)perspLeft[(i + (2 * frameW)) + 3]) +
                         (0.03 * (float)perspLeft[(i + (3 * frameW)) - 2]) + (0.03 * (float)perspLeft[(i + (3 * frameW)) - 1]) + (0.03 * (float)perspLeft[(i + (3 * frameW))]) + (0.03 * (float)perspLeft[(i + (3 * frameW)) + 1]) + (0.03 * (float)perspLeft[(i + (3 * frameW)) + 2]) + (0.03 * (float)perspLeft[(i + (3 * frameW)) + 3]))/255;
        float res  = midLeft - surLeft;
        if(index < 0) res = -1;
        else res  = midLeft - surLeft;

        // Picking out units of computation
        if (((currX) % rgcparams.divFactors[3] == 0) && ((currY) % rgcparams.divFactors[3] == 0)){

            // To check if the current index is in the middle,vertically
            if((currY >= (fovY - (fovWidthMidPre + paraLength + periLength))) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength)))){

                // -------------- Translating Indices ----------------
                //Left Section
                if(currX < fovX){
                    checkr = 1;
                    rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +
                           (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * (rgcparams.oz2side / rgcparams.divFactors[4])) +
                           (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * ((currX - (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side))) / rgcparams.divFactors[3]));
                }
                    // Right Section
                else {
                    checkr = 2;
                    rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (((currY > 0) && (currY < rgcparams.perspHeight)) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +

                            (int)((((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * (((currY >= ((double)rgcparams.oz3up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up)))) ? 1 : 0) * ((double)rgcparams.oz2side / (double)rgcparams.divFactors[4])) +
                                  (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength) + (2 * (double)rgcparams.oz1side)) / (double)rgcparams.divFactors[4]))) +


                            (int)((((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up)))) ? 1 : 0) * ((double)rgcparams.oz1side / (double)rgcparams.divFactors[3])) +
                                  (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength)) / (double)rgcparams.divFactors[3]))) +


                            (int)((((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * 2 * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)))) ? 1 : 0) * ((double)rgcparams.periLength / (double)rgcparams.divFactors[2])) +
                                  (((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength)) / (double)rgcparams.divFactors[2]))) +


                            (int)((((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * 2 * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)))) ? 1 : 0) * ((double)rgcparams.paraLength / (double)rgcparams.divFactors[1])) +
                                  (((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength + paraLength))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength + paraLength))))) ? 1 : 0) * (((double)rgcparams.foveaWidth) / (double)rgcparams.divFactors[1]))) +


                            (rgcparams.foveaWidth * (((currY >= (rgcparams.oz3up + rgcparams.oz2up + rgcparams.oz1up + periLength + paraLength)) && (currY < (rgcparams.perspHeight - (rgcparams.oz3up + rgcparams.oz2up + rgcparams.oz1up + periLength + paraLength)))) ? 1 : 0)) +
                           (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * ((currX - (fovX + (fovWidthMidPost + paraLength + periLength + 1))) / rgcparams.divFactors[3]));
                }
            }

                // Checks if current index is in the top or bottom sections
            else {
                rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +
                       (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * (rgcparams.oz2side / rgcparams.divFactors[4])) +
                       (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * ((currX - (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side))) / rgcparams.divFactors[3]));

            }

            // Entering data
            rgcLeftDevice[yMatch[currY]][rgcX] = rgcX;
        }
    }

    // ------------------------------------- X -------------------------------------

    // ------------------------------------------ OZ2 processing -------------------------------------

    if (// Inner Perimeters

        // Left
            ((currX < (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side))) && (currX >= (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side))) && (currY >= (fovY - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up))) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up)))) ||
            // Top
            ((currX >= (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side))) && (currX <= (fovX + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side))) && (currY >= (fovY - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up))) && (currY < (fovY - (fovWidthMidPre + paraLength  + periLength + rgcparams.oz1up)))) ||
            // Right
            ((currX > (fovX + fovWidthMidPost + paraLength + periLength + rgcparams.oz1side)) && (currX <= (fovX + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side))) && (currY >= (fovY - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up))) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up)))) ||
            // Bottom
            ((currX >= (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side))) && (currX <= (fovX + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side))) && (currY > (fovY + fovWidthMidPost + paraLength + periLength + rgcparams.oz1up)) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up))))){

        int RGCoffset = (fovWidth * fovWidth) + (((((fovWidth + (2 * paraLength)) / rgcparams.divFactors[1])) * ( paraLength / rgcparams.divFactors[1] )) + ((paraLength) * (fovWidth/rgcparams.divFactors[1])) + ((((fovWidth + (2 * paraLength)) / rgcparams.divFactors[1]) * (paraLength / rgcparams.divFactors[1])))) +
                        (((((fovWidth + (paraLength * 2) + (periLength * 2)) / rgcparams.divFactors[2]) * (periLength / rgcparams.divFactors[2])) + (((2 * periLength) / rgcparams.divFactors[2]) * ((fovWidth + (2 * paraLength)) / rgcparams.divFactors[2]))) + ((((fovWidth + (paraLength * 2) + (periLength * 2)) / rgcparams.divFactors[2])) * (periLength / rgcparams.divFactors[2]))) +
                        (((fovWidth + (paraLength * 2) + (periLength * 2) + (rgcparams.oz1side * 2)) / rgcparams.divFactors[3]) * (rgcparams.oz1up / rgcparams.divFactors[3])) + ((((rgcparams.oz1side / rgcparams.divFactors[3]) * 2)) * ((fovWidth + (2 * paraLength) + (2 * periLength)) / rgcparams.divFactors[3])) +
                        ((((fovWidth + (paraLength * 2) + (periLength * 2) + (rgcparams.oz1side * 2)) / rgcparams.divFactors[3])) * (rgcparams.oz1up / rgcparams.divFactors[3]));

        int sectionWidth = 0;
        int currentSum = 0;
        int checkr = 0;

        float midLeft = (0.11111111 * (((float)perspLeft[i - frameW - 1])/255)) + (0.11111111 * (((float)perspLeft[i - frameW])/255)) + (0.11111111 * (((float)perspLeft[i - frameW + 1])/255)) +
                        (0.11111111 * (((float)perspLeft[i - 1])/255)) + (0.11111111 * (((float)perspLeft[i])/255)) + (0.11111111 * (((float)perspLeft[i + 1])/255)) +
                        (0.11111111 * (((float)perspLeft[i + frameW - 1])/255)) + (0.11111111 * (((float)perspLeft[i + frameW])/255)) + (0.11111111 * (((float)perspLeft[i + frameW + 1])/255));
        float surLeft = (
                                // center - 4 layer (Row 1)
                                (0.013889 * (float)perspLeft[(i - (4 * frameW)) - 4]) + (0.013889 * (float)perspLeft[(i - (4 * frameW)) - 3]) + (0.013889 * (float)perspLeft[(i - (4 * frameW)) - 2]) + (0.013889 * (float)perspLeft[(i - (4 * frameW)) - 1]) + (0.013889 * (float)perspLeft[(i - (4 * frameW))]) +
                                (0.013889 * (float)perspLeft[(i - (4 * frameW)) + 1]) + (0.013889 * (float)perspLeft[(i - (4 * frameW)) + 2]) + (0.013889 * (float)perspLeft[(i - (4 * frameW)) + 3]) + (0.013889 * (float)perspLeft[(i - (4 * frameW)) + 4]) +
                                // center - 3 layer (Row 2)
                                (0.013889 * (float)perspLeft[(i - (3 * frameW)) - 4]) + (0.013889 * (float)perspLeft[(i - (3 * frameW)) - 3]) + (0.013889 * (float)perspLeft[(i - (3 * frameW)) - 2]) + (0.013889 * (float)perspLeft[(i - (3 * frameW)) - 1]) + (0.013889 * (float)perspLeft[(i - (3 * frameW))]) +
                                (0.013889 * (float)perspLeft[(i - (3 * frameW)) + 1]) + (0.013889 * (float)perspLeft[(i - (3 * frameW)) + 2]) + (0.013889 * (float)perspLeft[(i - (3 * frameW)) + 3]) + (0.013889 * (float)perspLeft[(i - (3 * frameW)) + 4]) +
                                // center - 2 layer (Row 3)
                                (0.013889 * (float)perspLeft[(i - (2 * frameW)) - 4]) + (0.013889 * (float)perspLeft[(i - (2 * frameW)) - 3]) + (0.013889 * (float)perspLeft[(i - (2 * frameW)) - 2]) + (0.013889 * (float)perspLeft[(i - (2 * frameW)) - 1]) + (0.013889 * (float)perspLeft[(i - (2 * frameW))]) +
                                (0.013889 * (float)perspLeft[(i - (2 * frameW)) + 1]) + (0.013889 * (float)perspLeft[(i - (2 * frameW)) + 2]) + (0.013889 * (float)perspLeft[(i - (2 * frameW)) + 3]) + (0.013889 * (float)perspLeft[(i - (2 * frameW)) + 4]) +
                                // center - 1 layer (Row 4)
                                (0.013889 * (float)perspLeft[(i - (1 * frameW)) - 4]) + (0.013889 * (float)perspLeft[(i - (1 * frameW)) - 3]) + (0.013889 * (float)perspLeft[(i - (1 * frameW)) - 2]) +
                                (0.013889 * (float)perspLeft[(i - (1 * frameW)) + 2]) + (0.013889 * (float)perspLeft[(i - (1 * frameW)) + 3]) + (0.013889 * (float)perspLeft[(i - (1 * frameW)) + 4]) +
                                // Center layer (Row 5)
                                (0.013889 * (float)perspLeft[i - 4]) + (0.013889 * (float)perspLeft[i  - 3]) + (0.013889 * (float)perspLeft[i - 2]) +
                                (0.013889 * (float)perspLeft[i  + 2]) + (0.013889 * (float)perspLeft[i + 3]) + (0.013889 * (float)perspLeft[i + 4]) +
                                // Center layer + 1 (Row 6)
                                (0.013889 * (float)perspLeft[(i + (1 * frameW)) - 4]) + (0.013889 * (float)perspLeft[(i + (1 * frameW)) - 3]) + (0.013889 * (float)perspLeft[(i + (1 * frameW)) - 2]) +
                                (0.013889 * (float)perspLeft[(i + (1 * frameW)) + 2]) + (0.013889 * (float)perspLeft[(i + (1 * frameW)) + 3]) + (0.013889 * (float)perspLeft[(i + (1 * frameW)) + 4]) +
                                // center + 2 layer (Row 7)
                                (0.013889 * (float)perspLeft[(i + (2 * frameW)) - 4]) + (0.013889 * (float)perspLeft[(i + (2 * frameW)) - 3]) + (0.013889 * (float)perspLeft[(i + (2 * frameW)) - 2]) + (0.013889 * (float)perspLeft[(i + (2 * frameW)) - 1]) + (0.013889 * (float)perspLeft[(i + (2 * frameW))]) +
                                (0.013889 * (float)perspLeft[(i + (2 * frameW)) + 1]) + (0.013889 * (float)perspLeft[(i + (2 * frameW)) + 2]) + (0.013889 * (float)perspLeft[(i + (2 * frameW)) + 3]) + (0.013889 * (float)perspLeft[(i + (2 * frameW)) + 4]) +
                                // center + 3 layer (Row 8)
                                (0.013889 * (float)perspLeft[(i + (3 * frameW)) - 4]) + (0.013889 * (float)perspLeft[(i + (3 * frameW)) - 3]) + (0.013889 * (float)perspLeft[(i + (3 * frameW)) - 2]) + (0.013889 * (float)perspLeft[(i + (3 * frameW)) - 1]) + (0.013889 * (float)perspLeft[(i + (3 * frameW))]) +
                                (0.013889 * (float)perspLeft[(i + (3 * frameW)) + 1]) + (0.013889 * (float)perspLeft[(i + (3 * frameW)) + 2]) + (0.013889 * (float)perspLeft[(i + (3 * frameW)) + 3]) + (0.013889 * (float)perspLeft[(i + (3 * frameW)) + 4]) +
                                // center - 4 layer (Row 9)
                                (0.013889 * (float)perspLeft[(i + (4 * frameW)) - 4]) + (0.013889 * (float)perspLeft[(i + (4 * frameW)) - 3]) + (0.013889 * (float)perspLeft[(i + (4 * frameW)) - 2]) + (0.013889 * (float)perspLeft[(i + (4 * frameW)) - 1]) + (0.013889 * (float)perspLeft[(i + (4 * frameW))]) +
                                (0.013889 * (float)perspLeft[(i + (4 * frameW)) + 1]) + (0.013889 * (float)perspLeft[(i + (4 * frameW)) + 2]) + (0.013889 * (float)perspLeft[(i + (4 * frameW)) + 3]) + (0.013889 * (float)perspLeft[(i + (4 * frameW)) + 4]))/255;
        float res  = midLeft - surLeft;
        if(index < 0) res = -1;
        else res  = midLeft - surLeft;

        int compIndex = 0;

        // Picking out units of computation
        if (((currX) % rgcparams.divFactors[4] == 0) && ((currY) % rgcparams.divFactors[4] == 0)){

            // To check if the current index is in the middle,vertically
            if((currY >= (fovY - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1up))) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1up)))){

                // -------------- Translating Indices ----------------
                //Left Section
                if(currX < fovX){
                    checkr = 1;
                    rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +
                           (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * ((currX - (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side))) / rgcparams.divFactors[4]));
                }
                    // Right Section
                else {
                    checkr = 2;
                    rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (((currY > 0) && (currY < rgcparams.perspHeight)) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +


                            (int)((((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * (((currY >= ((double)rgcparams.oz3up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up)))) ? 1 : 0) * ((double)rgcparams.oz2side / (double)rgcparams.divFactors[4])) +
                                  (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength) + (2 * (double)rgcparams.oz1side)) / (double)rgcparams.divFactors[4]))) +


                            (int)((((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * 2 * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up)))) ? 1 : 0) * ((double)rgcparams.oz1side / (double)rgcparams.divFactors[3])) +
                                  (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength)) / (double)rgcparams.divFactors[3]))) +


                            (int)((((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * 2 * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)))) ? 1 : 0) * ((double)rgcparams.periLength / (double)rgcparams.divFactors[2])) +
                                  (((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength)) / (double)rgcparams.divFactors[2]))) +


                            (int)((((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * 2 * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)))) ? 1 : 0) * ((double)rgcparams.paraLength / (double)rgcparams.divFactors[1])) +
                                  (((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength + paraLength))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength + paraLength))))) ? 1 : 0) * (((double)rgcparams.foveaWidth) / (double)rgcparams.divFactors[1]))) +


                           (rgcparams.foveaWidth * (((currY >= (rgcparams.oz3up + rgcparams.oz2up + rgcparams.oz1up + periLength + paraLength)) && (currY < (rgcparams.perspHeight - (rgcparams.oz3up + rgcparams.oz2up + rgcparams.oz1up + periLength + paraLength)))) ? 1 : 0)) +
                           (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * ((currX - (fovX + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1side + 1))) / rgcparams.divFactors[4]));
                }
            }

                // Checks if current index is in the top or bottom sections
            else {
                checkr = 2;
                rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +
                       (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * ((currX - (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side))) / rgcparams.divFactors[4]));

            }

            // Entering data
            rgcLeftDevice[yMatch[currY]][rgcX] = rgcX;
        }
    }

    // ------------------------------------------ OZ3 processing -------------------------------------

    if (// Inner Perimeters

        // Left
            ((currX < (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side))) && (currX >= (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side + rgcparams.oz3side))) && (currY >= (fovY - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up + rgcparams.oz3up))) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up + rgcparams.oz3up)))) ||
            // Top
            ((currX >= (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side + rgcparams.oz3side))) && (currX <= (fovX + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side + rgcparams.oz3side))) && (currY >= (fovY - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up + rgcparams.oz3up))) && (currY < (fovY - (fovWidthMidPre + paraLength  + periLength + rgcparams.oz1up + rgcparams.oz2up)))) ||
            // Right
            ((currX > (fovX + fovWidthMidPost + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side)) && (currX <= (fovX + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side + rgcparams.oz3side))) && (currY >= (fovY - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up + rgcparams.oz3up))) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up + rgcparams.oz3up)))) ||
            // Bottom
            ((currX >= (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side + rgcparams.oz3side))) && (currX <= (fovX + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side + rgcparams.oz3side))) && (currY > (fovY + fovWidthMidPost + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up)) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up + rgcparams.oz3up))))){

        int RGCoffset = (fovWidth * fovWidth) + (((((fovWidth + (2 * paraLength)) / rgcparams.divFactors[1])) * ( paraLength / 2 )) + ((paraLength) * (fovWidth/rgcparams.divFactors[1])) + ((((fovWidth + (2 * paraLength)) / rgcparams.divFactors[1]) * (paraLength / rgcparams.divFactors[1])))) +
                        (((((fovWidth + (paraLength * 2) + (periLength * 2)) / rgcparams.divFactors[2]) * (periLength / rgcparams.divFactors[2])) + (((2 * periLength) / rgcparams.divFactors[2]) * ((fovWidth + (2 * paraLength)) / rgcparams.divFactors[2]))) + ((((fovWidth + (paraLength * 2) + (periLength * 2)) / rgcparams.divFactors[2])) * (periLength / rgcparams.divFactors[2]))) +
                        (((fovWidth + (paraLength * 2) + (periLength * 2) + (rgcparams.oz1side * 2)) / rgcparams.divFactors[3]) * (rgcparams.oz1up / rgcparams.divFactors[3])) + ((((rgcparams.oz1side / rgcparams.divFactors[3]) * 2)) * ((fovWidth + (2 * paraLength) + (2 * periLength)) / rgcparams.divFactors[3])) +
                        ((((fovWidth + (paraLength * 2) + (periLength * 2) + (rgcparams.oz1side * 2)) / rgcparams.divFactors[3])) * (rgcparams.oz1up / rgcparams.divFactors[3])) + (((fovWidth + (paraLength * 2) + (periLength * 2) + (rgcparams.oz1side * 2) + (rgcparams.oz2side * 2)) / rgcparams.divFactors[4]) * (rgcparams.oz2up / rgcparams.divFactors[4])) +
                        ((((rgcparams.oz2side / rgcparams.divFactors[4]) * 2)) * ((fovWidth + (2 * paraLength) + (2 * periLength) + (2 * rgcparams.oz1up)) / rgcparams.divFactors[4])) + ((((fovWidth + (paraLength * 2) + (periLength * 2) + (rgcparams.oz1side * 2) + (rgcparams.oz2side * 2)) / rgcparams.divFactors[4])) * (rgcparams.oz2up / rgcparams.divFactors[4]));
        int sectionWidth = 0;
        int currentSum = 0;
        int checkr = 0;

        float midLeft = // Center - 2
                (0.04 * (((float)perspLeft[i - (2 * frameW) - 2])/255)) + (0.04 * (((float)perspLeft[i - (2 * frameW) - 1])/255)) + (0.04 * (((float)perspLeft[i - (2 * frameW)])/255)) + (0.04 * (((float)perspLeft[i - (2 * frameW) + 1])/255)) + (0.04 * (((float)perspLeft[i - (2 * frameW) + 2])/255)) +
                // Center - 1
                (0.04 * (((float)perspLeft[i - (1 * frameW) - 2])/255)) + (0.04 * (((float)perspLeft[i - frameW - 1])/255)) + (0.04 * (((float)perspLeft[i - frameW])/255)) + (0.04 * (((float)perspLeft[i - frameW + 1])/255)) + (0.04 * (((float)perspLeft[i - frameW + 2])/255)) +
                // Center
                (0.04 * (((float)perspLeft[i - 2])/255)) + (0.04 * (((float)perspLeft[i - 1])/255)) + (0.04 * (((float)perspLeft[i])/255)) + (0.04 * (((float)perspLeft[i + 1])/255)) + (0.04 * (((float)perspLeft[i + 2])/255)) +
                // Center + 1
                (0.04 * (((float)perspLeft[i + frameW - 2])/255)) + (0.04 * (((float)perspLeft[i + frameW - 1])/255)) + (0.04 * (((float)perspLeft[i + frameW])/255)) + (0.04 * (((float)perspLeft[i + frameW + 1])/255)) + (0.04 * (((float)perspLeft[i + frameW + 2])/255)) +
                // Center + 2
                (0.04 * (((float)perspLeft[i + (2 * frameW) - 2])/255)) + (0.04 * (((float)perspLeft[i + (2 * frameW) - 1])/255)) + (0.04 * (((float)perspLeft[i + (2 * frameW)])/255)) + (0.04 * (((float)perspLeft[i + (2 * frameW) + 1])/255)) + (0.04 * (((float)perspLeft[i + (2 * frameW) + 2])/255));
        float surLeft = (
                                // center - 6 layer (Row 1)
                                (0.006944 * (float)perspLeft[(i - (6 * frameW)) - 6]) + (0.006944 * (float)perspLeft[(i - (6 * frameW)) - 5]) + (0.006944 * (float)perspLeft[(i - (6 * frameW)) - 4]) + (0.006944 * (float)perspLeft[(i - (6 * frameW)) - 3]) + (0.006944 * (float)perspLeft[(i - (6 * frameW)) - 2]) + (0.006944 * (float)perspLeft[(i - (6 * frameW)) - 1]) + (0.006944 * (float)perspLeft[(i - (6 * frameW))]) +
                                (0.006944 * (float)perspLeft[(i - (6 * frameW)) + 1]) + (0.006944 * (float)perspLeft[(i - (6 * frameW)) + 2]) + (0.006944 * (float)perspLeft[(i - (6 * frameW)) + 3]) + (0.006944 * (float)perspLeft[(i - (6 * frameW)) + 4]) + (0.006944 * (float)perspLeft[(i - (6 * frameW)) + 5]) + (0.006944 * (float)perspLeft[(i - (6 * frameW)) + 6]) +
                                // center - 5 layer (Row 2)
                                (0.006944 * (float)perspLeft[(i - (5 * frameW)) - 6]) + (0.006944 * (float)perspLeft[(i - (5 * frameW)) - 5]) + (0.006944 * (float)perspLeft[(i - (5 * frameW)) - 4]) + (0.006944 * (float)perspLeft[(i - (5 * frameW)) - 3]) + (0.006944 * (float)perspLeft[(i - (5 * frameW)) - 2]) + (0.006944 * (float)perspLeft[(i - (5 * frameW)) - 1]) + (0.006944 * (float)perspLeft[(i - (5 * frameW))]) +
                                (0.006944 * (float)perspLeft[(i - (5 * frameW)) + 1]) + (0.006944 * (float)perspLeft[(i - (5 * frameW)) + 2]) + (0.006944 * (float)perspLeft[(i - (5 * frameW)) + 3]) + (0.006944 * (float)perspLeft[(i - (5 * frameW)) + 4]) + (0.006944 * (float)perspLeft[(i - (5 * frameW)) + 5]) + (0.006944 * (float)perspLeft[(i - (5 * frameW)) + 6]) +
                                // center - 4 layer (Row 3)
                                (0.006944 * (float)perspLeft[(i - (4 * frameW)) - 6]) + (0.006944 * (float)perspLeft[(i - (4 * frameW)) - 5]) + (0.006944 * (float)perspLeft[(i - (4 * frameW)) - 4]) + (0.006944 * (float)perspLeft[(i - (4 * frameW)) - 3]) + (0.006944 * (float)perspLeft[(i - (4 * frameW)) - 2]) + (0.006944 * (float)perspLeft[(i - (4 * frameW)) - 1]) + (0.006944 * (float)perspLeft[(i - (4 * frameW))]) +
                                (0.006944 * (float)perspLeft[(i - (4 * frameW)) + 1]) + (0.006944 * (float)perspLeft[(i - (4 * frameW)) + 2]) + (0.006944 * (float)perspLeft[(i - (4 * frameW)) + 3]) + (0.006944 * (float)perspLeft[(i - (4 * frameW)) + 4]) + (0.006944 * (float)perspLeft[(i - (4 * frameW)) + 5]) + (0.006944 * (float)perspLeft[(i - (4 * frameW)) + 6]) +
                                // center - 3 layer (Row 4)
                                (0.006944 * (float)perspLeft[(i - (3 * frameW)) - 6]) + (0.006944 * (float)perspLeft[(i - (3 * frameW)) - 5]) + (0.006944 * (float)perspLeft[(i - (3 * frameW)) - 4]) + (0.006944 * (float)perspLeft[(i - (3 * frameW)) - 3]) + (0.006944 * (float)perspLeft[(i - (3 * frameW)) - 2]) + (0.006944 * (float)perspLeft[(i - (3 * frameW)) - 1]) + (0.006944 * (float)perspLeft[(i - (3 * frameW))]) +
                                (0.006944 * (float)perspLeft[(i - (3 * frameW)) + 1]) + (0.006944 * (float)perspLeft[(i - (3 * frameW)) + 2]) + (0.006944 * (float)perspLeft[(i - (3 * frameW)) + 3]) + (0.006944 * (float)perspLeft[(i - (3 * frameW)) + 4]) + (0.006944 * (float)perspLeft[(i - (3 * frameW)) + 5]) + (0.006944 * (float)perspLeft[(i - (3 * frameW)) + 6]) +
                                // center - 2 layer (Row 5)
                                (0.006944 * (float)perspLeft[(i - (2 * frameW)) - 6]) + (0.006944 * (float)perspLeft[(i - (2 * frameW)) - 5]) + (0.006944 * (float)perspLeft[(i - (2 * frameW)) - 4]) + (0.006944 * (float)perspLeft[(i - (2 * frameW)) - 3]) +
                                (0.006944 * (float)perspLeft[(i - (2 * frameW)) + 3]) + (0.006944 * (float)perspLeft[(i - (2 * frameW)) + 4]) + (0.006944 * (float)perspLeft[(i - (2 * frameW)) + 5]) + (0.006944 * (float)perspLeft[(i - (2 * frameW)) + 6]) +
                                // center - 1 layer (Row 6)
                                (0.006944 * (float)perspLeft[(i - (1 * frameW)) - 6]) + (0.006944 * (float)perspLeft[(i - (1 * frameW)) - 5]) + (0.006944 * (float)perspLeft[(i - (1 * frameW)) - 4]) + (0.006944 * (float)perspLeft[(i - (1 * frameW)) - 3]) +
                                (0.006944 * (float)perspLeft[(i - (1 * frameW)) + 3]) + (0.006944 * (float)perspLeft[(i - (1 * frameW)) + 4]) + (0.006944 * (float)perspLeft[(i - (1 * frameW)) + 5]) + (0.006944 * (float)perspLeft[(i - (1 * frameW)) + 6]) +
                                // Center layer (Row 7)
                                (0.006944 * (float)perspLeft[i - 6]) + (0.006944 * (float)perspLeft[i - 5]) + (0.006944 * (float)perspLeft[i - 4]) + (0.006944 * (float)perspLeft[i  - 3]) +
                                (0.006944 * (float)perspLeft[i + 3]) + (0.006944 * (float)perspLeft[i + 4]) + (0.006944 * (float)perspLeft[i + 5]) + (0.006944 * (float)perspLeft[i + 6]) +
                                // Center layer + 1 (Row 8)
                                (0.006944 * (float)perspLeft[(i + (1 * frameW)) - 6]) + (0.006944 * (float)perspLeft[(i + (1 * frameW)) - 5]) + (0.006944 * (float)perspLeft[(i + (1 * frameW)) - 4]) + (0.006944 * (float)perspLeft[(i + (1 * frameW)) - 3]) +
                                (0.006944 * (float)perspLeft[(i + (1 * frameW)) + 3]) + (0.006944 * (float)perspLeft[(i + (1 * frameW)) + 4]) + (0.006944 * (float)perspLeft[(i + (1 * frameW)) + 5]) + (0.006944 * (float)perspLeft[(i + (1 * frameW)) + 6]) +
                                // center + 2 layer (Row 9)
                                (0.006944 * (float)perspLeft[(i + (2 * frameW)) - 6]) + (0.006944 * (float)perspLeft[(i + (2 * frameW)) - 5]) + (0.006944 * (float)perspLeft[(i + (2 * frameW)) - 4]) + (0.006944 * (float)perspLeft[(i + (2 * frameW)) - 3]) +
                                (0.006944 * (float)perspLeft[(i + (2 * frameW)) + 3]) + (0.006944 * (float)perspLeft[(i + (2 * frameW)) + 4]) + (0.006944 * (float)perspLeft[(i + (2 * frameW)) + 5]) + (0.006944 * (float)perspLeft[(i + (2 * frameW)) + 6]) +
                                // center + 3 layer (Row 10)
                                (0.006944 * (float)perspLeft[(i + (3 * frameW)) - 6]) + (0.006944 * (float)perspLeft[(i + (3 * frameW)) - 5]) + (0.006944 * (float)perspLeft[(i + (3 * frameW)) - 4]) + (0.006944 * (float)perspLeft[(i + (3 * frameW)) - 3]) + (0.006944 * (float)perspLeft[(i + (3 * frameW)) - 2]) + (0.006944 * (float)perspLeft[(i + (3 * frameW)) - 1]) + (0.006944 * (float)perspLeft[(i + (3 * frameW))]) +
                                (0.006944 * (float)perspLeft[(i + (3 * frameW)) + 1]) + (0.006944 * (float)perspLeft[(i + (3 * frameW)) + 2]) + (0.006944 * (float)perspLeft[(i + (3 * frameW)) + 3]) + (0.006944 * (float)perspLeft[(i + (3 * frameW)) + 4]) + (0.006944 * (float)perspLeft[(i + (3 * frameW)) + 5]) + (0.006944 * (float)perspLeft[(i + (3 * frameW)) + 6]) +
                                // center - 4 layer (Row 11)
                                (0.006944 * (float)perspLeft[(i + (4 * frameW)) - 6]) + (0.006944 * (float)perspLeft[(i + (4 * frameW)) - 5]) + (0.006944 * (float)perspLeft[(i + (4 * frameW)) - 4]) + (0.006944 * (float)perspLeft[(i + (4 * frameW)) - 3]) + (0.006944 * (float)perspLeft[(i + (4 * frameW)) - 2]) + (0.006944 * (float)perspLeft[(i + (4 * frameW)) - 1]) + (0.006944 * (float)perspLeft[(i + (4 * frameW))]) +
                                (0.006944 * (float)perspLeft[(i + (4 * frameW)) + 1]) + (0.006944 * (float)perspLeft[(i + (4 * frameW)) + 2]) + (0.006944 * (float)perspLeft[(i + (4 * frameW)) + 3]) + (0.006944 * (float)perspLeft[(i + (4 * frameW)) + 4]) + (0.006944 * (float)perspLeft[(i + (4 * frameW)) + 5]) + (0.006944 * (float)perspLeft[(i + (4 * frameW)) + 6]) +
                                // center + 5 layer (Row 12)
                                (0.006944 * (float)perspLeft[(i + (5 * frameW)) - 6]) + (0.006944 * (float)perspLeft[(i + (5 * frameW)) - 5]) + (0.006944 * (float)perspLeft[(i + (5 * frameW)) - 4]) + (0.006944 * (float)perspLeft[(i + (5 * frameW)) - 3]) + (0.006944 * (float)perspLeft[(i + (5 * frameW)) - 2]) + (0.006944 * (float)perspLeft[(i + (5 * frameW)) - 1]) + (0.006944 * (float)perspLeft[(i + (5 * frameW))]) +
                                (0.006944 * (float)perspLeft[(i + (5 * frameW)) + 1]) + (0.006944 * (float)perspLeft[(i + (5 * frameW)) + 2]) + (0.006944 * (float)perspLeft[(i + (5 * frameW)) + 3]) + (0.006944 * (float)perspLeft[(i + (5 * frameW)) + 4]) + (0.006944 * (float)perspLeft[(i + (5 * frameW)) + 5]) + (0.006944 * (float)perspLeft[(i + (5 * frameW)) + 6]) +
                                // center - 6 layer (Row 13)
                                (0.006944 * (float)perspLeft[(i + (6 * frameW)) - 6]) + (0.006944 * (float)perspLeft[(i + (6 * frameW)) - 5]) + (0.006944 * (float)perspLeft[(i + (6 * frameW)) - 4]) + (0.006944 * (float)perspLeft[(i + (6 * frameW)) - 3]) + (0.006944 * (float)perspLeft[(i + (6 * frameW)) - 2]) + (0.006944 * (float)perspLeft[(i + (6 * frameW)) - 1]) + (0.006944 * (float)perspLeft[(i + (6 * frameW))]) +
                                (0.006944 * (float)perspLeft[(i + (6 * frameW)) + 1]) + (0.006944 * (float)perspLeft[(i + (6 * frameW)) + 2]) + (0.006944 * (float)perspLeft[(i + (6 * frameW)) + 3]) + (0.006944 * (float)perspLeft[(i + (6 * frameW)) + 4]) + (0.006944 * (float)perspLeft[(i + (6 * frameW)) + 5]) + (0.006944 * (float)perspLeft[(i + (6 * frameW)) + 6]))/255;
        float res  = midLeft - surLeft;
        if(index < 0) res = -1;
        else res  = midLeft - surLeft;

        int compIndex = 0;

        // Picking out units of computation
        if (((currX) % rgcparams.divFactors[5] == 0) && ((currY) % rgcparams.divFactors[5] == 0)){


            // To check if the current index is in the middle,vertically
            if((currY >= (fovY - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up))) && (currY <= (fovY + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1up + rgcparams.oz2up)))){



                // -------------- Translating Indices ----------------
                //Left Section
                if(currX < fovX){
                    checkr = 4;
                    rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * ((currX - (fovX - (fovWidthMidPre + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side + rgcparams.oz3side))) / rgcparams.divFactors[5]));
                }
                    // Right Section
                else {
                    checkr = 5;
                    rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * (((currY > 0) && (currY < rgcparams.perspHeight)) ? 1 : 0) * (rgcparams.oz3side / rgcparams.divFactors[5])) +

                            // side series (that's why we multiply by 2)
                            (int)((((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * 2 * (((currY >= ((double)rgcparams.oz3up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up)))) ? 1 : 0) * ((double)rgcparams.oz2side / (double)rgcparams.divFactors[4])) +
                            // mid series
                           (((currY % rgcparams.divFactors[4] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength) + (2 * (double)rgcparams.oz1side)) / (double)rgcparams.divFactors[4]))) +

                            (int)((((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * 2 * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up)))) ? 1 : 0) * ((double)rgcparams.oz1side / (double)rgcparams.divFactors[3])) +
                           (((currY % rgcparams.divFactors[3] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength)) / (double)rgcparams.divFactors[3]))) +

                            (int)((((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * 2 * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)))) ? 1 : 0) * ((double)rgcparams.periLength / (double)rgcparams.divFactors[2])) +
                           (((currY % rgcparams.divFactors[2] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))))) ? 1 : 0) * (((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength)) / (double)rgcparams.divFactors[2]))) +

                            (int)((((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * 2 * (((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)) && (currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)))) ? 1 : 0) * ((double)rgcparams.paraLength / (double)rgcparams.divFactors[1])) +
                           (((currY % rgcparams.divFactors[1] == 0) ? 1 : 0) * ((((currY >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength)) & (currY < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength + paraLength))) || ((currY < ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength))) & (currY >= ((double)rgcparams.perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + periLength + paraLength))))) ? 1 : 0) * (((double)rgcparams.foveaWidth) / (double)rgcparams.divFactors[1]))) +

                           (rgcparams.foveaWidth * (((currY >= (rgcparams.oz3up + rgcparams.oz2up + rgcparams.oz1up + periLength + paraLength)) && (currY < (rgcparams.perspHeight - (rgcparams.oz3up + rgcparams.oz2up + rgcparams.oz1up + periLength + paraLength)))) ? 1 : 0)) +
                           (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * ((currX - (fovX + (fovWidthMidPost + paraLength + periLength + rgcparams.oz1side + rgcparams.oz2side + 1))) / rgcparams.divFactors[5]));
                    // if(currY == 350 & currX == 3080) rgcLeftDevice[0][0] = rgcX;
                }

            }

                // Checks if current index is in the top or bottom sections
            else {
                checkr = 6;
                rgcX = (((currY % rgcparams.divFactors[5] == 0) ? 1 : 0) * ((currX) / rgcparams.divFactors[5]));

            }

            // Entering data
            rgcLeftDevice[yMatch[currY]][rgcX] = rgcX;
        }
    }

    // ------------------------------------- X -------------------------------------

}

__global__
void eye1Pipeline(int foveaPoint, int pixelCount, ::uint8_t  *a, ::uint8_t *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < 1000) a[i] = 22.89;

}


RGCdev visualPass1 (){


    // Video processing parameters
    VideoReaderState vr_stateLeft, vr_stateRight;
    if (!video_reader_open(&vr_stateLeft, "/home/kasm-user/CLionProjects/BlackTest/src/cud/beachLeft.mp4")) {
        cout << "ERROR!!" << endl;
        cout << "Couldn't open video file (make sure you set a video file that exists" << endl;
    }
    if (!video_reader_open(&vr_stateRight, "/home/kasm-user/CLionProjects/BlackTest/src/cud/beachRight.mp4")) {
        cout << "ERROR!!" << endl;
        cout << "Couldn't open video file (make sure you set a video file that exists" << endl;
    }

    // Allocate frameLeft buffer
    constexpr int ALIGNMENT = 128;
    const int frame_width = vr_stateLeft.width;
    const int frame_height = vr_stateLeft.height;
    const int perspHeight = 2160;
    const int perspWidth = 3840;
    int numOfPixels = frame_width * frame_height;
    int *retinaDivs;

    GLFWwindow* window;
    int frCount = 0 ;

    if (!glfwInit()) {
        printf("Couldn't init GLFW\n");
    }

    window = glfwCreateWindow(1280, 720, "Hello World", NULL, NULL);
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
        rgcparams.foveaWidth = 180;
        rgcparams.divFactors[0] = 1; // 32,400 - 129,600
        rgcparams.paraLength = 90;
        rgcparams.divFactors[1] = 2; // 45,000 - 90,000
        rgcparams.periLength = 210;
        rgcparams.divFactors[2] = 3; // 34,000
        rgcparams.oz1up = 120;
        rgcparams.oz1side = 255;
        rgcparams.divFactors[3] = 3; // 60,000 122400-corners 336600-sides 158400-tops
        rgcparams.oz2up = 250;
        rgcparams.oz2side = 510;
        rgcparams.divFactors[4] = 5; // 81,000
        rgcparams.oz3up = 320;
        rgcparams.oz3side = 765;
        rgcparams.divFactors[5] = 7; // 83,686
    }

    rgcparams.perspHeight = perspHeight;
    rgcparams.perspWidth = perspWidth;
    //------ Init - VAriables required to calculate RGC inputs in formRGCinputs -----------
    int *yMatchHost, *xWidthsHost, *yMatchDev, *xWidthsDev, rgcHeight, tmpxSum, rgcArrayHeight = -1;
    yMatchHost = (int *)malloc(perspHeight * sizeof(int));
    xWidthsHost = (int *)malloc(perspHeight * sizeof(int));
    //------------------------------------------------------------------
    for(int i = 0; i < perspHeight; i+=1){
        tmpxSum =
                // ------------------------------------ Side series ----------------------------------

                // - OZ3
                (int)(((((i % rgcparams.divFactors[5] == 0) && (i >= 0 && i < perspHeight)) ? 1 : 0) * (2 * ((double)rgcparams.oz3side / (double)rgcparams.divFactors[5]))) +

                      ((((i % rgcparams.divFactors[5] == 0) && ((i >= 0 && i < (double)rgcparams.oz3up) || (i >= (perspHeight - (double)rgcparams.oz3up) && i < perspHeight))) ? 1 : 0) * ((((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength) + (2 * (double)rgcparams.oz1side) + (2 * (double)rgcparams.oz2side)) / (double)rgcparams.divFactors[5])))) +

                // - OZ2
                (int)(((((i % rgcparams.divFactors[4] == 0) && (i >= (double)rgcparams.oz3up && i < (perspHeight - (double)rgcparams.oz3up))) ? 1 : 0) * (2 * ((double)rgcparams.oz2side / (double)rgcparams.divFactors[4]))) +

                      ((((i % rgcparams.divFactors[4] == 0) && ((i >= (double)rgcparams.oz3up && i < ((double)rgcparams.oz3up + (double)rgcparams.oz2up)) || (i >= (perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up)) && i < (perspHeight - (double)rgcparams.oz3up)))) ? 1 : 0) * ((((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength) + (2 * (double)rgcparams.oz1side)) / (double)rgcparams.divFactors[4])))) +

                // - OZ1
                (int)(((((i % rgcparams.divFactors[3] == 0) && (i >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up) && i < (perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up)))) ? 1 : 0) * (2 * ((double)rgcparams.oz1side / (double)rgcparams.divFactors[3]))) +

                      ((((i % rgcparams.divFactors[3] == 0) && ((i >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up) && i < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)) || (i >= (perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)) && i < (perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up))))) ? 1 : 0) * ((((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength) + (2 * (double)rgcparams.periLength)) / (double)rgcparams.divFactors[3])))) +

                // - Peri
                (int)(((((i % rgcparams.divFactors[2] == 0) && (i >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up) && i < (perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up)))) ? 1 : 0) * (2 * ((double)rgcparams.periLength / (double)rgcparams.divFactors[2]))) +

                      ((((i % rgcparams.divFactors[2] == 0) && ((i >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up) && i < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + (double)rgcparams.periLength)) || (i >= (perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + (double)rgcparams.periLength)) && i < (perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up))))) ? 1 : 0) * ((((double)rgcparams.foveaWidth + (2 * (double)rgcparams.paraLength)) / (double)rgcparams.divFactors[2])))) +

                // - Para
                (int)(((((i % rgcparams.divFactors[1] == 0) && (i >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + (double)rgcparams.periLength) && i < (perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + (double)rgcparams.periLength)))) ? 1 : 0) * (2 * ((double)rgcparams.paraLength / (double)rgcparams.divFactors[1]))) +

                      ((((i % rgcparams.divFactors[1] == 0) && ((i >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + (double)rgcparams.periLength) && i < ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + (double)rgcparams.periLength + (double)rgcparams.paraLength)) || (i >= (perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + (double)rgcparams.periLength + (double)rgcparams.paraLength)) && i < (perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + (double)rgcparams.periLength))))) ? 1 : 0) * ((((double)rgcparams.foveaWidth) / (double)rgcparams.divFactors[1])))) +

                // - Fovea
                ((((i % rgcparams.divFactors[0] == 0) && (i >= ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + (double)rgcparams.periLength + (double)rgcparams.paraLength) && i < (perspHeight - ((double)rgcparams.oz3up + (double)rgcparams.oz2up + (double)rgcparams.oz1up + (double)rgcparams.periLength + (double)rgcparams.paraLength)))) ? 1 : 0) * (double)rgcparams.foveaWidth);

        if(tmpxSum != 0){
            rgcArrayHeight+=1;
            xWidthsHost[rgcArrayHeight] = tmpxSum;
        }
        yMatchHost[i] = rgcArrayHeight;
    }
    rgcArrayHeight += 1;

    //------- Prep - the 2D arrays needed to capture RGC input ----------------
    float **rgcLeftHost, **rgcRightHost, **rgcLeftDev, **rgcRightDev, **rgcPin;
    rgcLeftHost = (float**)malloc(rgcArrayHeight * sizeof(float*));
    rgcRightHost = (float**)malloc(rgcArrayHeight * sizeof(float*));
    rgcPin = (float**)malloc(rgcArrayHeight * sizeof(float*));
    cudaMalloc(&rgcLeftDev, rgcArrayHeight * sizeof(float*));
    cudaMalloc(&rgcRightDev, rgcArrayHeight * sizeof(float*));
    cudaMalloc(&xWidthsDev, perspHeight * sizeof(int));
    cudaMalloc(&yMatchDev, perspHeight * sizeof(int));

    for(int i = 0; i < rgcArrayHeight; i+=1){
        rgcPin[i] = (float*)malloc(xWidthsHost[i] * sizeof(float));
        cudaMalloc((void **)&rgcLeftHost[i], xWidthsHost[i] * sizeof(float));
        cudaMalloc((void **)&rgcRightHost[i], xWidthsHost[i] * sizeof(float));
    }

    cudaMemcpy(rgcLeftDev, rgcLeftHost, rgcArrayHeight * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(rgcRightDev, rgcRightHost, rgcArrayHeight * sizeof(float*), cudaMemcpyHostToDevice);
    // -------------------------------------------------------------------


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
    int u = 0, p_y = 0, p_uv = 0, corrID = 0;
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
    params.perspfov = 90;
    params.transform = NULL;
    params.ntransform = 0;
    params.debug = false;

    // --------------------------------------------------------

     //----------------- Sample transformation --------------------------
    params.transform = static_cast<TRANSFORM *>(realloc(params.transform,
                                                        (params.ntransform + 1) * sizeof(TRANSFORM)));
    params.transform[params.ntransform].axis = ZPAN;
    params.transform[params.ntransform].value = (M_PI / 180)*(-60);
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
    ::uint8_t *currentFrameLeft, *currentFrameRight, *pinnedTemp;
    uint8_t *perspHost, *perspLeft, *perspRight, *persp1;
    PARAMS *testparams;
    FRUSTUM testFrust;
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
    cudaMalloc(&worldLeft, numOfPixels * 4 * sizeof(::uint8_t));
    cudaMalloc(&worldRight, numOfPixels * 4 * sizeof(::uint8_t));
    cudaMalloc(&perspLeft, (perspHeight * perspWidth) * 4 * sizeof(uint8_t));
    cudaMalloc(&perspRight, (perspHeight * perspWidth) * 4 * sizeof(uint8_t));
    cudaMalloc(&persp1, (perspHeight * perspWidth) * 4 * sizeof(uint8_t));
    cudaMalloc(&devTrans, params.ntransform * sizeof(TRANSFORM));
    cudaMalloc(&retinaDivs, 6 * sizeof(int));
    cudaSetDevice(1);
    cudaStreamCreate(&memStream2);
    cudaStreamCreate ( &funcStream2);

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
    for (int fr = 0; fr < 9 ; fr+=1){
        frameLeft = video_reader_read_frame(&vr_stateLeft, frame_data_left, &pts);
        u = 0, p_y = 0, p_uv = 0, corrID = 0;
        for (int y = 0; y < frameLeft->height; y++){
            for (int x = 0; x < frame_width; x++){
                p_y = (y * width) + x;

                corrID = ((y/2) * frameLeft->linesize[2]) + (x / 2);

                // R
                dataLeft[u] = frameLeft->data[0][p_y] + (1.370705 * (frameLeft->data[2][corrID] - 128));
                dataRight[u] = frameRight->data[0][p_y] + (1.370705 * (frameRight->data[2][corrID] - 128));

                // G
                dataLeft[u + 1] = frameLeft->data[0][p_y] - (0.337633 * (frameLeft->data[1][corrID] - 128)) - (0.698001 * (frameLeft->data[2][corrID] - 128));
                dataRight[u + 1] = frameRight->data[0][p_y] - (0.337633 * (frameRight->data[1][corrID] - 128)) - (0.698001 * (frameRight->data[2][corrID] - 128));

                // B
                dataLeft[u + 2] = frameLeft->data[0][p_y] + 1.732446 * (frameLeft->data[1][corrID] - 128);
                dataRight[u + 2] = frameRight->data[0][p_y] + 1.732446 * (frameRight->data[1][corrID] - 128);

                // A
                dataLeft[u + 3] = frameLeft->data[0][p_y];
                dataRight[u + 3] = frameRight->data[0][p_y];

                u+=4;
            }
        }
        frameArrayLeft.push_back(dataLeft);
        frameArrayRight.push_back(dataRight);
    }

    // ---------------------------------------------------------------------------------



    int frameIndex = 0;

    // --------------- Initial prep ----------------------
    cudaSetDevice(0);
    cudaMemcpy(devTrans, params.transform, params.ntransform * sizeof(TRANSFORM), cudaMemcpyHostToDevice);
    cudaMemcpy(retinaDivs, rgcparams.divFactors, 6 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(yMatchDev, yMatchHost, perspHeight * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(xWidthsDev, xWidthsHost, perspHeight * sizeof(int), cudaMemcpyHostToDevice);
    ::memcpy(currentFrameLeft, frameArrayLeft[7], numOfPixels * 4 * sizeof(::uint8_t));
    ::memcpy(currentFrameRight, frameArrayRight[7], numOfPixels * 4 * sizeof(::uint8_t));

    cudaDeviceSynchronize();
    rgcparams.divFactors = retinaDivs;
    params.transform = devTrans;

    int frameStorageCountr = 0;
    // Starts clock
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(worldLeft, currentFrameLeft, numOfPixels * 4 * sizeof(::uint8_t), cudaMemcpyHostToDevice, memStream1);
    cudaMemcpyAsync(worldRight, currentFrameRight, numOfPixels * 4 * sizeof(::uint8_t), cudaMemcpyHostToDevice, memStream2);
    cudaDeviceSynchronize();

    // Converts vr 360 video into a perspective frame of dimensions perspHeight x perspWidth
    world2Persp<<<((perspHeight * perspWidth) + 1023)/1024, 1024, 0, funcStream1>>>(worldLeft, perspLeft, worldRight, perspRight, params, frustum);
    cudaDeviceSynchronize();

    // Forms RGC inputs
    formRGCinputs<<<((perspHeight * perspWidth) + 1023)/1024, 1024, 0, funcStream1>>>(4145280, perspHeight, perspWidth, rgcparams, perspLeft, perspRight, rgcLeftDev, rgcRightDev, xWidthsDev, yMatchDev);
    cudaDeviceSynchronize();

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();

    // Stops clock
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<chrono::microseconds>(stop - start).count();
    cout << "Net duration of visual pass : " << duration << endl;

    // Transfers computed values back to host
    cudaSetDevice(0);

    // Transfer perspective frame back after computation - not necessary
    cudaMemcpy(perspHost, perspRight, (perspHeight * perspWidth * 4 ) * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Transfers RGC inputs back - required if doing neural computation on another GPU
//    cudaMemcpy(rgcLeftHost, rgcLeftDev, rgcArrayHeight * sizeof(float*), cudaMemcpyDeviceToHost);
//    for(int p = 0; p < rgcArrayHeight; p+=1){
//        cudaMemcpy(rgcPin[p], rgcLeftHost[p], xWidthsHost[p] * sizeof(float), cudaMemcpyDeviceToHost);
//
//        for(int j = 0; j < xWidthsHost[p]; j+=1){
//            if(rgcPin[p][j] != j){
//                cout << "Error at : " << p << " Index of Error is : " << j << endl;
//            }
//        }
////        cout << "i : " << p << " || Length : " << xWidthsHost[p] << " || ";
////        for(int j = 0; j < xWidthsHost[p]; j+=1){
////            cout << rgcPin[p][j] << " - ";
////        }
////        cout << endl;
//    }

    // cout << (int)hostTestr[5] << endl;


    glBindTexture(GL_TEXTURE_2D, tex_handle);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, perspWidth, perspHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, perspHost);

    // Render whatever you want
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex_handle);
    glBegin(GL_QUADS);
    glTexCoord2d(0,0); glVertex2i(0, 0);
    glTexCoord2d(1,0); glVertex2i(0 + 1280, 0);
    glTexCoord2d(1,1); glVertex2i(0 + 1280, 0 + 720);
    glTexCoord2d(0,1); glVertex2i(0, 0 + 720);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    glfwSwapBuffers(window);
    glfwPollEvents();
    //::getchar();

    rgcdev.parvo = rgcLeftDev;
    return rgcdev;
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
