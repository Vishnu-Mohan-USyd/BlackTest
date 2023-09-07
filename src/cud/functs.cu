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
void saxpy(RGC** RGCdet)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;



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

//    if(((x / 20) % 2 == 0) && ((y / 20) % 2 == 0)){
//        perspLeft[(i * 4)] = 0;
//        perspLeft[(i * 4) + 1] = 0;
//        perspLeft[(i * 4) + 2] = 0;
//        perspLeft[(i * 4) + 3] = 0;
//    } else {
//        perspLeft[(i * 4)] = 255;
//        perspLeft[(i * 4) + 1] = 255;
//        perspLeft[(i * 4) + 2] = 255;
//        perspLeft[(i * 4) + 3] = 255;
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
void formRGCcurrents(RGCPARAMS rgcparams, uint8_t *perspTest, uint8_t  *perspLeft, uint8_t  *perspRight, float** leftInputs, float** rightInputs, int* xWidths, int* yMatch, RGC** RGCdet)
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

    float surrSumL = 0, surrSumR = 0, cenSumL = 0, cenSumR = 0, surrIdeal = 0, cenIdeal = 0, surrValL = 0, cenValL = 0,
            surrValR = 0, cenValR = 0, xComp, yComp;
    int midX, midY, surrSide = (RGCdet[posY][posX].cenRfSide + (2 * RGCdet[posY][posX].surRfWidth)), currIndex, cenIndex, testr = 0;
    if(surrSide % 2 == 0) {
        midX = surrSide / 2;
        midY = midX;
    }
    if(surrSide % 2 == 1) {
        midX = (surrSide / 2) + 1;
        midY = midX;
    }

    cenIndex = (RGCdet[posY][posX].perspY * 4 * rgcparams.perspWidth) + (RGCdet[posY][posX].perspX * 4);

    for (int y = 1; y < (surrSide) + 1; y+=1){
        // Y boundary condition
        if ((((RGCdet[posY][posX].perspY < midY) && (y < midY) && ((midY - y) > RGCdet[posY][posX].perspY)) ||
             ((((rgcparams.perspHeight - 1) - (RGCdet[posY][posX].perspY)) < midY) && (y > midY) &&
              ((y - midY) > ((rgcparams.perspHeight - 1) - (RGCdet[posY][posX].perspY)))))) {
            continue;
        }
        for(int x = 1; x < (surrSide) + 1; x += 1){
            // X boundary condition
            if ((((RGCdet[posY][posX].perspX < midX) && (x < midX) && ((midX - x) > RGCdet[posY][posX].perspX)) ||
                 ((((rgcparams.perspWidth - 1) - (RGCdet[posY][posX].perspX)) < midX) && (x > midX) &&
                  ((x - midX) > ((rgcparams.perspWidth - 1) - (RGCdet[posY][posX].perspX)))))) {
                continue;
            }
            currIndex = cenIndex + ((y - midY) * 4 * rgcparams.perspWidth) + ((x - midX) * 4);
            xComp = 0; yComp = 0;

            // --------------------------- Surround Region -------------------------------
            if((x <= (RGCdet[posY][posX].surRfWidth)) || (x > (surrSide - RGCdet[posY][posX].surRfWidth)) ||
               (y <= (RGCdet[posY][posX].surRfWidth)) || (y > (surrSide - RGCdet[posY][posX].surRfWidth))){
                xComp = (float)x;
                yComp = (float)y;
                if(x > midX){
                    xComp = (float)((surrSide + 1) - x);
                }
                if(y > midY){
                    yComp = (float)((surrSide + 1) - y);
                }
                if (RGCdet[posY][posX].detType == LUM){
                    testr+=1;
                    surrSumL += (float)(xComp + yComp) * (float)perspLeft[currIndex + 3];
                    surrSumR += (float)(xComp + yComp) * (float)perspLeft[currIndex + 3];
                    surrIdeal += (float)(xComp + yComp) * 255;
                } else if (RGCdet[posY][posX].detType == COLOR){
                    if(RGCdet[posY][posX].colID == R_rgc){
                        // Surround is -M                     // M
                        surrSumL += ((xComp + yComp) * (float)perspLeft[currIndex + 1]);
                        surrSumR += ((xComp + yComp) * (float)perspRight[currIndex + 1]);
                        surrIdeal += (xComp + yComp) * 255;
                    } else if (RGCdet[posY][posX].colID == G_rgc){
                        // Surround is -(S + L)                                                 // S                              // L
                        surrSumL += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspLeft[currIndex + 2] : (float)perspLeft[currIndex]));
                        surrSumR += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspRight[currIndex + 2] : (float)perspRight[currIndex]));
                        surrIdeal += (xComp + yComp) * 255;
                    } else if(RGCdet[posY][posX].colID == B_rgc){
                        // Surround is -L                     // L
                        surrSumL += ((xComp + yComp) * (float)perspLeft[currIndex]);
                        surrSumR += ((xComp + yComp) * (float)perspRight[currIndex]);
                        surrIdeal += (xComp + yComp) * 255;
                    } else if (RGCdet[posY][posX].colID == Y_rgc){
                        // Surround is -(S + M)                                                 // S                              // M
                        surrSumL += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspLeft[currIndex + 2] : (float)perspLeft[currIndex + 1]));
                        surrSumR += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspRight[currIndex + 2] : (float)perspRight[currIndex + 1]));
                        surrIdeal += (xComp + yComp) * 255;
                    }
                }
            }
                // ----------------------------- X -------------------------------
                // --------------------------- Center Region -------------------------------
            else {
                xComp = (float)x - RGCdet[posY][posX].surRfWidth;
                yComp = (float)y - RGCdet[posY][posX].surRfWidth;
                if(x > midX){
                    xComp = (float)((RGCdet[posY][posX].cenRfSide + RGCdet[posY][posX].surRfWidth + 1) - x);
                }
                if(y > midY){
                    yComp = (float)((RGCdet[posY][posX].cenRfSide + RGCdet[posY][posX].surRfWidth + 1) - y);
                }
                if (RGCdet[posY][posX].detType == LUM){

                    cenSumL += (float)(xComp + yComp) * (float)perspLeft[currIndex + 3];
                    cenSumR += (float)(xComp + yComp) * (float)perspLeft[currIndex + 3];
                    cenIdeal += (float)(xComp + yComp) * 255;
                } else if (RGCdet[posY][posX].detType == COLOR){
                    if(RGCdet[posY][posX].colID == R_rgc){
                        // Center is (S+L)                                                     // S                              // L
                        cenSumL += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspLeft[currIndex + 2] : (float)perspLeft[currIndex]));
                        cenSumR += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspRight[currIndex + 2] : (float)perspRight[currIndex]));
                        cenIdeal += (xComp + yComp) * 255;
                    } else if (RGCdet[posY][posX].colID == G_rgc){
                        // Center is M                       // M
                        cenSumL += ((xComp + yComp) * (float)perspLeft[currIndex + 1]);
                        cenSumR += ((xComp + yComp) * (float)perspRight[currIndex + 1]);
                        cenIdeal += (xComp + yComp) * 255;
                    } else if(RGCdet[posY][posX].colID == B_rgc){
                        // Center is (S+M)                                                     // S                              // M
                        cenSumL += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspLeft[currIndex + 2] : (float)perspLeft[currIndex + 1]));
                        cenSumR += ((xComp + yComp) * (((x % 2 == 0) && (y % 2 == 0)) ? (float)perspRight[currIndex + 2] : (float)perspRight[currIndex + 1]));
                        cenIdeal += (xComp + yComp) * 255;
                    } else if (RGCdet[posY][posX].colID == Y_rgc){
                        // Center is L                       // L
                        cenSumL += ((xComp + yComp) * (float)perspLeft[currIndex]);
                        cenSumR += ((xComp + yComp) * (float)perspRight[currIndex]);
                        cenIdeal += (xComp + yComp) * 255;
                    }
                }
            }
            // ----------------------------- X -------------------------------
        }
    }
    // Calculating surround averages
    surrValL = surrSumL / surrIdeal;
    surrValR = surrSumR / surrIdeal;
    // Calculating center averages
    cenValL = cenSumL / cenIdeal;
    cenValR = cenSumR / cenIdeal;

    //if()

    if(RGCdet[posY][posX].type == MIDGET){
        perspTest[RGCdet[posY][posX].perspID] =  (int)(255);
        perspTest[RGCdet[posY][posX].perspID + 1] = (int)(255);
        perspTest[RGCdet[posY][posX].perspID + 2] =  (int)(255);
        perspTest[RGCdet[posY][posX].perspID + 3] = (int)(255);
    }
    // To rectify -ve responses, just multiply the below with
    // (((cenValR - surrValR) < 0) ? -0 : 1)
    leftInputs[posY][posX] = RGCdet[posY][posX].perspX;
    rightInputs[posY][posX] = (cenValR - surrValR);

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

void vid2rgc (int* frameNum, queue<float**> *rgcQueue_l, queue<float**> *rgcQueue_r, queue<RGC**> *rgcDetsQ, vid2rgcParams *v2rp, mutex &rgcMut, condition_variable &rgcCond){


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

    if (!glfwInit()) {
        printf("Couldn't init GLFW\n");
    }

    window = glfwCreateWindow(1920, 1080, "Hello World", NULL, NULL);
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
            OZ3ellipse = tillOZ3 + 800,
            OZ2ellipse = tillOZ2 + 400;

    //------ Init - VAriables required to calculate RGC inputs in initRGCdets -----------
    int *yMatchHost, *xWidthsHost, *yMatchDev, *xWidthsDev, tmpxSum, rgcArrayHeight = -1, RGCcount = 0, rgcInd = 0,
            fovy = ((perspHeight / 2) - 1), fovx = ((perspWidth / 2) - 1), diffX = 0, diffY = 0, divR = 0, mostProb = 0, ElXGr = 0, ElYGr = 0, ElXLs = 0, ElYLs = 0, divSector = 0;
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
    divLength = 0,
    // angle between point and origin
    angle = 0, angleDeg = 0;
    RGC** tmpRGCdets = (RGC**) malloc(perspHeight * sizeof(RGC*));
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
                    tmpRGCdets[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                }
                tmpRGCdets[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                currRand = uni(rng);
                if(currRand > 0.95){
                    tmpRGCdets[rgcArrayHeight][tmpxSum].type = PARASOL;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 6;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 8;
                } else {
                    currRand = uni(rng);
                    tmpRGCdets[rgcArrayHeight][tmpxSum].type = MIDGET;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 2;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 2;
                    currRand = uni(rng);
                    if(currRand < 0.25) {
                        currRand = uni(rng);
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = COLOR;
                        if(currRand < 0.4) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = R_rgc;
                        if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = G_rgc;
                        if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = B_rgc;
                        if(currRand >= 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                    }
                }
                tmpRGCdets[rgcArrayHeight][tmpxSum].perspX = j;
                tmpRGCdets[rgcArrayHeight][tmpxSum].perspY = i;
                tmpRGCdets[rgcArrayHeight][tmpxSum].zone = FOV;
                tmpxSum+=1;
            }
            // ----------------------- X --------------------------------

            // ---------------------- PARAFOVEAL REGION --------------------

            if(SOL < pow(tillPara, 2) && SOL >= pow(rgcparams.foveaWidth, 2)){

                // First, we find out just how far between the two bands the current radius is
                radProg = (((double)(SOL - pow(rgcparams.foveaWidth, 2)) / (double)(pow(tillPara, 2) - pow(rgcparams.foveaWidth, 2))));

                // Then we calculate the length of the subdivisions within the band
                divLength = 1 / (float)((rgcparams.divFactors[1] - rgcparams.divFactors[0]));
                mostProb = (int)(radProg * (float)((rgcparams.divFactors[1] - rgcparams.divFactors[0]) + 1));

                // If it's the last band
                if(radProg > 1 - divLength) {
                    if (currRand < ((radProg - (1 - divLength)) / divLength)) divR = rgcparams.divFactors[1];
                    else divR = rgcparams.divFactors[1] - 1;
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector = (int)(radProg / divLength);

                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector * divLength)) / divLength)) divR = rgcparams.divFactors[0] + divSector;

                        // if currRand is greater
                    else divR = rgcparams.divFactors[0] + divSector + 1;
                }

                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.95){
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 8;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 10;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 3;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 3;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].zone = PARA;
                    tmpxSum+=1;
                }
            }
            // ----------------------- X --------------------------

            // ---------------------- PERIFOVEAL REGION --------------------

            if(SOL < pow(tillPeri, 2) && SOL >= pow(tillPara, 2)){

                // First, we find out just how far between the two bands the current radius is
                radProg = (((double)(SOL - pow(tillPara, 2)) / (double)(pow(tillPeri, 2) - pow(tillPara, 2))));

                // Then we calculate the length of the subdivisions within the band
                divLength = 1 / (float)((rgcparams.divFactors[2] - rgcparams.divFactors[1]));

                // If it's the last band
                if(radProg > 1 - divLength) {
                    if (currRand < ((radProg - (1 - divLength)) / divLength)) divR = rgcparams.divFactors[2];
                    else divR = rgcparams.divFactors[2] - 1;
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector = (int)(radProg / divLength);

                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector * divLength)) / divLength)) divR = rgcparams.divFactors[1] + divSector;

                        // if currRand is greater
                    else divR = rgcparams.divFactors[1] + divSector + 1;
                }
                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.9){
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 12;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 14;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 2;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 3;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].zone = PERI;
                    tmpxSum+=1;
                }

            }
            // ----------------------- X --------------------------

            // ---------------------- OZ1 REGION --------------------

            if(SOL < pow(tillOZ1, 2) && SOL >= pow(tillPeri, 2)){

                // First, we find out just how far between the two bands the current radius is
                radProg = (((double)(SOL - pow(tillPeri, 2)) / (double)(pow(tillOZ1, 2) - pow(tillPeri, 2))));

                // Then we calculate the length of the subdivisions within the band
                divLength = 1 / (float)((rgcparams.divFactors[3] - rgcparams.divFactors[2]));


                // If it's the last band
                if(radProg > 1 - divLength) {
                    if (currRand < ((radProg - (1 - divLength)) / divLength)) divR = rgcparams.divFactors[3];
                    else divR = rgcparams.divFactors[3] - 1;
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector = (int)(radProg / divLength);

                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector * divLength)) / divLength)) divR = rgcparams.divFactors[2] + divSector;

                        // if currRand is greater
                    else divR = rgcparams.divFactors[2] + divSector + 1;
                }

                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.8){
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 15;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 18;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 4;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 5;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].zone = OZ1;
                    tmpxSum+=1;
                }

            }
            // ----------------------- X --------------------------

            // ---------------------- OZ2 REGION - Circular Side  --------------------

            if((SOL < pow(tillOZ2, 2) && (diffX < 0)) && SOL >= pow(tillOZ1, 2)){

                // First, we find out just how far between the two bands the current radius is
                radProg = (((double)(SOL - pow(tillOZ1, 2)) / (double)(pow(tillOZ2, 2) - pow(tillOZ1, 2))));

                // Then we calculate the length of the subdivisions within the band
                divLength = 1 / (float)((rgcparams.divFactors[4] - rgcparams.divFactors[3]));

                // If it's the last band
                if(radProg > 1 - divLength) {
                    if (currRand < ((radProg - (1 - divLength)) / divLength)) divR = rgcparams.divFactors[4];
                    else divR = rgcparams.divFactors[4] - 1;
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector = (int)(radProg / divLength);

                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector * divLength)) / divLength)) divR = rgcparams.divFactors[3] + divSector;

                        // if currRand is greater
                    else divR = rgcparams.divFactors[3] + divSector + 1;
                }
                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.7){
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 24;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 24;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 12;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 20;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].zone = OZ2;
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
                divLength = 1 / (float)((rgcparams.divFactors[4] - rgcparams.divFactors[3]));

                // If it's the last band
                if(radProg > 1 - divLength) {
                    divSector = (int)(radProg / divLength);
                    if (currRand < ((radProg - (1 - divLength)) / divLength)) divR = rgcparams.divFactors[4];
                    else divR = rgcparams.divFactors[4] - 1;
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector = (int)(radProg / divLength);

                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector * divLength)) / divLength)) divR = rgcparams.divFactors[3] + divSector;

                        // if currRand is greater
                    else divR = rgcparams.divFactors[3] + divSector + 1;
                }
                 cout << "SOL : " << SOL << " || SOLGr : " << SOLGr << " || SolLs : " << SOLLs << " || ElXLs : " << ElXLs <<
                                        " || tan : "  << tan(angle) << " || ElYLs : " << ElYLs << " || x : " << j - fovx << " || y :" << diffY << " || angle : " << angleDeg << " || divL : " <<
                                        divLength <<  " || sector : " << divSector << " || radProg : " << radProg << " || divR : " << divR << endl;
                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.7){
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 24;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 24;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 12;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 20;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].zone = OZ2;
                    tmpxSum+=1;
                }

            }
            // ----------------------- X --------------------------

            // ---------------------- OZ3 REGION - Circular side--------------------

            if(((SOL < pow(tillOZ3, 2)) && (diffX < 0)) && SOL >= pow(tillOZ2, 2)){

                // First, we find out just how far between the two bands the current radius is
                radProg = (((double)(SOL - pow(tillOZ2, 2)) / (double)(pow(tillOZ3, 2) - pow(tillOZ2, 2))));

                // Then we calculate the length of the subdivisions within the band
                divLength = 1 / (float)((rgcparams.divFactors[5] - rgcparams.divFactors[4]));

                // If it's the last band
                if(radProg > 1 - divLength) {
                    if (currRand < ((radProg - (1 - divLength)) / divLength)) divR = rgcparams.divFactors[5];
                    else divR = rgcparams.divFactors[5] - 1;
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector = (int)(radProg / divLength);

                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector * divLength)) / divLength)) divR = rgcparams.divFactors[4] + divSector;

                        // if currRand is greater
                    else divR = rgcparams.divFactors[4] + divSector + 1;
                }
                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.6){
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 36;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 48;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 20;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 20;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].zone = OZ3;
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
                divLength = 1 / (float)((rgcparams.divFactors[5] - rgcparams.divFactors[4]));

                // If it's the last band
                if(radProg > 1 - divLength) {
                    divSector = (int)(radProg / divLength);
                    if (currRand < ((radProg - (1 - divLength)) / divLength)) divR = rgcparams.divFactors[5];
                    else divR = rgcparams.divFactors[5] - 1;
                }
                    // all other bands
                else {

                    // Finding out which sector of the band radius has progressed till
                    divSector = (int)(radProg / divLength);


                    // If currRand is lesser than the radius's reach in the sector
                    if (currRand > ((radProg - ((float)divSector * divLength)) / divLength)) divR = rgcparams.divFactors[4] + divSector;

                        // if currRand is greater
                    else divR = rgcparams.divFactors[4] + divSector + 1;
                }
                if(i % divR == 0 && j % divR == 0){
                    if(tmpxSum == 0){
                        rgcArrayHeight+=1;
                        tmpRGCdets[rgcArrayHeight] = (RGC*)malloc(perspWidth * sizeof(RGC));
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspID = ((i * perspWidth * 4) + (j * 4));
                    currRand = uni(rng);
                    if(currRand > 0.6){
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = PARASOL;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 36;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 48;
                    } else {
                        currRand = uni(rng);
                        tmpRGCdets[rgcArrayHeight][tmpxSum].type = MIDGET;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].detType = LUM;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].cenRfSide = 20;
                        tmpRGCdets[rgcArrayHeight][tmpxSum].surRfWidth = 20;
                        currRand = uni(rng);
                        if(currRand < 0.25) {
                            currRand = uni(rng);
                            tmpRGCdets[rgcArrayHeight][tmpxSum].detType = COLOR;
                            if(currRand < 0.4) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = R_rgc;
                            if(currRand >= 0.4 && currRand < 0.7) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = G_rgc;
                            if(currRand >= 0.7 && currRand < 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = B_rgc;
                            if(currRand >= 0.9) tmpRGCdets[rgcArrayHeight][tmpxSum].colID = Y_rgc;
                        }
                    }
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspX = j;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].perspY = i;
                    tmpRGCdets[rgcArrayHeight][tmpxSum].zone = OZ3;
                    tmpxSum+=1;
                }
            }
            // ----------------------- X --------------------------
        }


        if(tmpxSum != 0){
            xWidthsHost[rgcArrayHeight] = tmpxSum;
            RGCcount += tmpxSum;
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
    RGC** RGCdets = (RGC**) malloc(rgcArrayHeight * sizeof(RGC*)), ** RGCdetsPin = (RGC**) malloc(rgcArrayHeight * sizeof(RGC*)), **RGCDetsDev, **detsArray_h, **detsArray_d;
    float **rgcInputsLeft_h, **rgcInputsRight_h, **rgcInputsLeft_d, **rgcInputsRight_d, **rgcInputPin_l, **rgcInputPin_r;
    detsArray_h = (RGC**)malloc(rgcArrayHeight * sizeof(RGC*));
    rgcInputsLeft_h = (float**)malloc(rgcArrayHeight * sizeof(float*));
    rgcInputsRight_h = (float**)malloc(rgcArrayHeight * sizeof(float*));
    rgcInputPin_l = (float**)malloc(rgcArrayHeight * sizeof(float*));
    rgcInputPin_r = (float**)malloc(rgcArrayHeight * sizeof(float*));
    cudaMalloc(&RGCDetsDev, rgcArrayHeight * sizeof(RGC*));
    cudaMalloc(&rgcInputsLeft_d, rgcArrayHeight * sizeof(float*));
    cudaMalloc(&rgcInputsRight_d, rgcArrayHeight * sizeof(float*));
    cudaMalloc(&xWidthsDev, perspHeight * sizeof(int));
    cudaMalloc(&yMatchDev, perspHeight * sizeof(int));

    for(int i = 0; i < rgcArrayHeight; i+=1){

        cudaMalloc((void**) &RGCdets[i], ((xWidthsHost[i]*sizeof(RGC))));
        cudaMemcpy(RGCdets[i], tmpRGCdets[i], xWidthsHost[i] * sizeof(RGC), cudaMemcpyHostToDevice);
        RGCdetsPin[i] = (RGC*) malloc(xWidthsHost[i] * sizeof(RGC));
        rgcInputPin_l[i] = (float*)malloc(xWidthsHost[i] * sizeof(float));
        cudaMalloc((void **)&rgcInputsLeft_h[i], xWidthsHost[i] * sizeof(float));
        cudaMalloc((void **)&rgcInputsRight_h[i], xWidthsHost[i] * sizeof(float));
    }

    cudaMemcpy(RGCDetsDev,RGCdets,rgcArrayHeight * sizeof(RGC*),cudaMemcpyHostToDevice);
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
    params.transform[params.ntransform].value = (M_PI / 180)*(60);
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

    while(true){


        frameLeft = video_reader_read_frame(&vr_stateLeft, frame_data_left, &pts);
        frameRight = video_reader_read_frame(&vr_stateRight, frame_data_right, &pts);
        cudaMemcpy(ffmpegLY, frameLeft->data[0], numOfPixels * sizeof(::uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ffmpegRY, frameRight->data[0], numOfPixels * sizeof(::uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ffmpegLU, frameLeft->data[1], (numOfPixels / 4) * sizeof(::uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ffmpegRU, frameRight->data[1], (numOfPixels / 4) * sizeof(::uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ffmpegLV, frameLeft->data[2], (numOfPixels / 4) * sizeof(::uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ffmpegRV, frameRight->data[2], (numOfPixels / 4) * sizeof(::uint8_t), cudaMemcpyHostToDevice);
        cudaSetDevice(0);




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


        // Changes the ffmpeg YUV frames to RGB WorldFrames
        ffmpeg2World <<<(numOfPixels + 1023)/1024, 1024, 0, funcStream1>>> (worldLeft, worldRight, ffmpegLY, ffmpegRY,
                                                                            ffmpegLU, ffmpegRU, ffmpegLV, ffmpegRV, params);
        cudaDeviceSynchronize();

        // Converts vr 360 video into a perspective frame of dimensions perspHeight x perspWidth
        world2Persp<<<((perspHeight * perspWidth) + 1023)/1024, 1024, 0, funcStream1>>>(worldLeft, perspLeft, worldRight, perspRight, params, frustum);
        cudaDeviceSynchronize();

        // Creates test frame
        createPerspTest<<<((perspHeight * perspWidth) + 1023)/1024, 1024, 0, funcStream1>>>(perspTest, perspLeft, params);
        cudaDeviceSynchronize();


        // Forms RGC inputs
        formRGCcurrents<<<(RGCcount + 1023) / 1024, 1024, 0, funcStream1>>>(rgcparams, perspTest, perspLeft, perspRight,
                                                                            rgcInputsLeft_d, rgcInputsRight_d,
                                                                            xWidthsDev, yMatchDev, RGCDetsDev);
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
            cudaMemcpy(RGCdets, RGCDetsDev, rgcArrayHeight * sizeof(RGC*), cudaMemcpyDeviceToHost);
            for(int p = 0; p < rgcArrayHeight; p+=1){
                cudaMemcpy(RGCdetsPin[p], RGCdets[p], xWidthsHost[p] * sizeof(RGC), cudaMemcpyDeviceToHost);
            }
            rgcDetsQ->push(RGCdetsPin);
        }


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

//        auto stop = std::chrono::high_resolution_clock::now();
//        auto duration = std::chrono::duration_cast<chrono::microseconds>(stop - start).count();
//        cout << "Net duration of visual pass : " << duration << endl;



        glBindTexture(GL_TEXTURE_2D, tex_handle);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, perspWidth, perspHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, perspHost);

        // Render whatever you want
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex_handle);
        glBegin(GL_QUADS);
        glTexCoord2d(0,0); glVertex2i(0, 0);
        glTexCoord2d(1,0); glVertex2i(0 + 1920, 0);
        glTexCoord2d(1,1); glVertex2i(0 + 1920, 0 + 1080);
        glTexCoord2d(0,1); glVertex2i(0, 0 + 1080);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        glfwSwapBuffers(window);
        glfwPollEvents();
        // ::getchar();
        toIgnore+=1;
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
