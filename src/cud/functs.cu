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

using namespace std;
__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (n -1)) x[i] = x[i+1] ;

}

__global__
void formRGCinputs(int foveaPoint, int pixelCount, int frameH, int frameW, ::uint8_t  *Y, ::uint8_t *U, ::uint8_t *V, float *rgc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Here we aim to divide i into different retinal compartments
     * Fovea - 300 x 300 || Total RGCs are 90,000
     * Parafovea - 200 plus on all sides || Total RGCs are 100,150
     * Perifovea - 300 plus on all sides || Total RGCs are 133,200
     * */

    int fovY = foveaPoint/frameW;
    int fovX = foveaPoint - (fovY * frameW);
    int currY = i/frameW;
    int currX = i - (currY * frameW);
    // rgc[i] = 2;

    // ----------------------- Foveal Processing -----------------------
    if((currX >= fovX - 149) && (currY >= fovY - 149) && (currX <= fovX + 150) && (currY <= fovY + 150)){
        // rgc[2] = 1;
        int RGCoffset = 0;
        int sectionWidth = 300;
        float c1 = ((float)Y[i])/255;
        float c2 = ((0.125 * (float)Y[(i - frameW) - 1]) + (0.125 * (float)Y[(i - frameW)]) + (0.125 * (float)Y[(i - frameW) + 1]) +
                    (0.125 * (float)Y[(i) - 1]) + (0.125 * (float)Y[(i) + 1]) +
                    (0.125 * (float)Y[(i + frameW) - 1]) + (0.125 * (float)Y[(i + frameW)]) + (0.125 * (float)Y[(i + frameW) + 1]))/255;
        float res  = c1 - c2;
        if(res < 0) res = 0;

        // Converting from 2D modelled frame to linear RGC array
        rgc[((currX - (fovX - 149)) + ((currY - (fovY - 149))  * sectionWidth))] = ((currX - (fovX - 149)) + ((currY - (fovY - 149))  * sectionWidth));

        // rgc[((currX - (fovX - 149)) + ((currY - (fovY - 149))  * sectionWidth))] = (float)Y[(i - frameW) + 1];
    }

        // ---------------------- Parafoveal Processing --------------------

    if (// Inner Perimeters

        // Left
            ((currX <= (fovX - 149)) && (currX >= (fovX - (149 + 200))) && (currY >= (fovY - (149 + 200))) && (currY <= (fovY + (150 + 200)))) ||
            // Top
            ((currX >= (fovX - (149 + 200))) && (currX <= (fovX + (150 + 200))) && (currY >= (fovY - (149 + 200))) && (currY <= (fovY - 149))) ||
            // Right
            ((currX >= (fovX + 150)) && (currX <= (fovX + (150 + 200))) && (currY >= (fovY - (149 + 200))) && (currY <= (fovY + (150 + 200)))) ||
            // Bottom
            ((currX >= (fovX - (149 + 200))) && (currX <= (fovX + (150 + 200))) && (currY >= (fovY + 150)) && (currY <= (fovY + (150 + 200))))){

        //rgc[90001] = 2;
        int RGCoffset = 90000;
        int sectionWidth = 0;
        int currentSum = 0;
        int checkr = 0;
//        if (((currY >= (fovY - (149 + 200))) && (currY <= (fovY - 149))) ||
//                ((currX >= (fovX - (149 + 200))) && (currX <= (fovX + (150 + 200))) && (currY >= (fovY + 150)) && (currY <= (fovY + (149 + 200)))))
//            sectionWidth = 350;
//        else if ((currY >= (fovY - (149))) && (currY <= (fovY + (149))))
//            sectionWidth = 201;

        float c1 = ((float)Y[i])/255;
        float c2 = ((0.125 * (float)Y[(i - frameW) - 1]) + (0.125 * (float)Y[(i - frameW)]) + (0.125 * (float)Y[(i - frameW) + 1]) +
                    (0.125 * (float)Y[(i) - 1]) + (0.125 * (float)Y[(i) + 1]) +
                    (0.125 * (float)Y[(i + frameW) - 1]) + (0.125 * (float)Y[(i + frameW)]) + (0.125 * (float)Y[(i + frameW) + 1]))/255;
        float res  = c1 - c2;
        // rgc[90001] = 2;
        // The ParaComputationalUnits;
        int compIndex = 0;

        // Picking out units of computation
        if (((currX % 2) == (fovX % 2)) && ((currY % 2) == (fovY % 2))){

            // To check if the current index is in the middle
            if((currY >= (fovY - (149))) && (currY <= (fovY + (150)))){
                sectionWidth = 201;
                currentSum = 350 * 100;
                int rightAdder = 100;

                // -------------- Translating Indices ----------------
                //Left Section
                if(currX < fovX){
                    checkr = 1;
                    compIndex = ((currX - (fovX - (349)))/2 + ((currY - (fovY - (149)))/2 * sectionWidth)) + RGCoffset + currentSum;
                }
                    // Right Section
                else {
                    checkr = 2;
                    compIndex = (rightAdder + (currX - (fovX + 150))/2 + ((currY - (fovY - (149)))/2 * sectionWidth)) + RGCoffset + currentSum;
                }
            }

                // Checks if current index is in the top or bottom sections
            else {
                // Top Section
                if(currY < fovY){
                    checkr = 3;
                    sectionWidth = 350;
                    compIndex = ((currX - (fovX - (349)))/2 + (((currY - (fovY - (349)))/2) * sectionWidth)) + RGCoffset;
                }
                    // Bottom Section
                else {
                    checkr = 4;
                    sectionWidth = 350;
                    currentSum = (350 * 100) + (201 * 150);
                    compIndex = ((currX - (fovX - (349)))/2 + ((currY - (fovY + (151)))/2 * sectionWidth)) + RGCoffset + currentSum;
                }

                //--------------------------X--------------------------

            }

            // Entering data
            rgc[compIndex] = compIndex;
        }
    }
        // ---------------------------------- X -----------------------------------


        //------------------------ Perifoveal Processing -------------------------
    if (// Inner Perimeters

        // Left
            ((currX <= (fovX - (149 + 200))) && (currX >= (fovX - (149 + 200 + 300))) && (currY >= (fovY - (149 + 200 + 300))) && (currY <= (fovY + (150 + 200 + 300)))) ||
            // Top
            ((currX >= (fovX - (149 + 200 + 300))) && (currX <= (fovX + (150 + 200 + 300))) && (currY >= (fovY - (149 + 200 + 300))) && (currY <= (fovY - (149 + 200)))) ||
            // Right
            ((currX >= (fovX + 150 + 200)) && (currX <= (fovX + (150 + 200 + 300))) && (currY >= (fovY - (149 + 200 + 300))) && (currY <= (fovY + (150 + 200 + 300)))) ||
            // Bottom
            ((currX >= (fovX - (149 + 200 + 300))) && (currX <= (fovX + (150 + 200 + 300))) && (currY >= (fovY + 150 + 200)) && (currY <= (fovY + (150 + 200 + 300))))){
        int RGCoffset = 190150;
        int sectionWidth = 0;
        int currentSum = 0;
        int checkr = 0;

        float c1 = (0.25 * (((float)Y[i])/255)) + (0.25 * (((float)Y[i + 1])/255)) + (0.25 * (((float)Y[i + frameW])/255)) + (0.25 * (((float)Y[i + frameW + 1])/255));
        float c2 = ((0.08 * (float)Y[(i - frameW) - 1]) + (0.08 * (float)Y[(i - frameW)]) + (0.08 * (float)Y[(i - frameW) + 1]) + (0.08 * (float)Y[(i - frameW) + 1]) + (0.08 * (float)Y[(i - frameW) + 2]) +
                    (0.08 * (float)Y[(i) - 1]) + (0.08 * (float)Y[(i) + 2]) + (0.08 * (float)Y[(i) + frameW - 1]) + (0.08 * (float)Y[(i) + frameW + 2]) +
                    (0.08 * (float)Y[(i + (2 * frameW)) - 1]) + (0.08 * (float)Y[(i + (2 * frameW))]) + (0.08 * (float)Y[(i + (2 * frameW)) + 1]) + (0.08 * (float)Y[(i + (2 * frameW)) + 1]))/255;
        float res  = c1 - c2;

        int compIndex = 0;

        // Picking out units of computation
        if (((currX - (fovX - 649)) % 3 == 1) && ((currY - (fovY - 649)) % 3 == 1)){

            // To check if the current index is in the middle
            if((currY >= (fovY - (149 + 200))) && (currY <= (fovY + (150 + 200)))){
                sectionWidth = 200;
                currentSum = 433 * 100;
                int rightAdder = 100;

                // -------------- Translating Indices ----------------
                //Left Section
                if(currX < fovX){
                    checkr = 1;
                    compIndex = (((currX - (fovX - (649))))/3 + (((currY - (fovY - (349))))/3 * sectionWidth)) + RGCoffset + currentSum;
                }
                    // Right Section
                else {
                    checkr = 2;
                    compIndex = (rightAdder + ((currX - (fovX + 350)))/3 + (((currY - (fovY - (349))))/3 * sectionWidth)) + RGCoffset + currentSum;
                }
            }

                // Checks if current index is in the top or bottom sections
            else {
                // Top Section
                if(currY < fovY){
                    checkr = 3;
                    sectionWidth = 433;
                    compIndex = (((currX - (fovX - (649))))/3 + ((((currY - (fovY - (649))))/3) * sectionWidth)) + RGCoffset;
                }
                    // Bottom Section
                else {
                    checkr = 4;
                    sectionWidth = 433;
                    currentSum = (433 * 100) + (200 * 233);
                    compIndex = (((currX - (fovX - (649))))/3 + (((currY - (fovY + (350))))/3 * sectionWidth)) + RGCoffset + currentSum;
                }

                //--------------------------X--------------------------

            }

            // Entering data
            rgc[compIndex] = compIndex;
        }
    }



}

__global__
void eye1Pipeline(int foveaPoint, int pixelCount, ::uint8_t  *a, ::uint8_t *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < 1000) a[i] = 22.89;

}

//__global__
//void arrInit(int n, float a, float *x, float *y)
//{
//    int i = blockIdx.x*blockDim.x + threadIdx.x;
//    y[i] = a*x[i] + y[i];
//}



int visualPass1 (){

    // Video processing parameters
    VideoReaderState vr_state;
    if (!video_reader_open(&vr_state, "/home/kasm-user/CLionProjects/BlackTest/src/cud/sing.mp4")) {
        cout << "ERROR!!" << endl;
        cout << "Couldn't open video file (make sure you set a video file that exists" << endl;
    }

    // Allocate frame buffer
    constexpr int ALIGNMENT = 128;
    const int frame_width = vr_state.width;
    const int frame_height = vr_state.height;
    int numOfPixels = frame_width * frame_height;
    uint8_t* frame_data;
    cout << frame_height << endl;
    printf("\x1B[34m                         \tWidth : \033[0m"); cout << frame_width << endl;
    if (posix_memalign((void**)&frame_data, ALIGNMENT, frame_width * frame_height * 4) != 0) {
        cout << "ERROR!!" << endl;
        printf("Couldn't allocate frame buffer\n");
    }


    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    float *rgcCurrents, *hostTest;
    hostTest = (float*)malloc(numOfPixels*sizeof(float));
    ::uint8_t *Y_1, *U_1, *V_1;
    ::uint8_t *frameHolderHost, *pinnedTemp;

    cudaDeviceProp a{};
    cudaSetDevice(0);
    cudaGetDeviceProperties(&a, 0);

    std::cout << "Device Overlap :         " << a.deviceOverlap << endl;

    cudaStream_t memStream1, memStream2, funcStream1, funcStream2 ;

    cudaSetDevice(0);
    cudaStreamCreate ( &memStream1) ;
    cudaStreamCreate ( &funcStream1) ;
    cudaMalloc(&Y_1, numOfPixels*sizeof(::uint8_t));
    cudaMalloc(&rgcCurrents, numOfPixels*sizeof(float)); // RGC Currents are held here
    // cudaMalloc(&frameHolderHost, numOfPixels*sizeof(float ));
    cudaMalloc(&U_1, numOfPixels*sizeof(::uint8_t));
    cudaMalloc(&V_1, numOfPixels*sizeof(::uint8_t));
    cudaSetDevice(1);
    cudaStreamCreate(&memStream2);
    cudaStreamCreate ( &funcStream2);

    // Begin main loop

    frameHolderHost = (uint8_t*)malloc(numOfPixels * sizeof(uint8_t));

    int64_t pts;
    AVFrame* frame = video_reader_read_frame(&vr_state, frame_data, &pts);
    const unsigned int bytes = frame_height * frame_width * sizeof(uint8_t);
    cudaMallocHost((void**)&frameHolderHost, bytes);
    cudaMallocHost((void**)&hostTest, numOfPixels * sizeof(float));
    vector<::uint8_t *> frameStorage;
    for (int fr = 0; fr < 9 ; fr+=1){
        frame = video_reader_read_frame(&vr_state, frame_data, &pts);
        frameStorage.push_back(frame->data[0]);
        cout << fr << endl;
    }
    // ::memcpy(frameHolderHost, frameStorage[0], numOfPixels * sizeof(::uint8_t));
    int frameIndex = 0;
    // cout << (int) frameStorage[5][8] << endl;

    // --------------- Initial prep ----------------------
    cudaSetDevice(0);
    cudaMemcpy(Y_1, frameHolderHost, numOfPixels*sizeof(::uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(U_1, frameHolderHost, numOfPixels*sizeof(::uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(V_1, frameHolderHost, numOfPixels*sizeof(::uint8_t), cudaMemcpyHostToDevice);
    ::memcpy(frameHolderHost, frameStorage[7], numOfPixels*sizeof(::uint8_t));
//            cudaMemcpyAsync(V_2, frame->data[2], numOfPixels*sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaDeviceSynchronize();

// -----------------------------------------


    // std::vector<std::thread> threads;


    float * y1;
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < 1000; i+=1){

        if(i%36 == 1){
            ::memcpy(frameHolderHost, frameStorage[7], numOfPixels*sizeof(::uint8_t));
            // Transfers data from Video array to local pinned memory
            // cudaMemcpyAsync(frameHolderHost, frameStorage[0], numOfPixels * sizeof(::uint8_t), cudaMemcpyHostToHost, memStream1);
            cudaSetDevice(0);
            cudaMemcpyAsync(Y_1, frameHolderHost, numOfPixels*sizeof(::uint8_t), cudaMemcpyHostToDevice, memStream1);
            cudaMemcpyAsync(U_1, frameHolderHost, numOfPixels*sizeof(::uint8_t), cudaMemcpyHostToDevice, memStream1);
            cudaMemcpyAsync(V_1, frameHolderHost, numOfPixels*sizeof(::uint8_t), cudaMemcpyHostToDevice, memStream1);
//            cudaSetDevice(0);
//            cudaMemcpyAsync(frameHolderHost, frameStorage[0], numOfPixels * sizeof(::uint8_t), cudaMemcpyHostToHost, memStream1);
        }
        if(i%36 == 35){
            cudaSetDevice(0);
            cudaDeviceSynchronize();
            formRGCinputs<<<(numOfPixels + 1023)/1024, 1024, 0, funcStream1>>>(16592640, numOfPixels, frame_height, frame_width, Y_1, U_1, V_1, rgcCurrents);
            cudaDeviceSynchronize();
        }

        // eye1Pipeline<<<(numOfPixels + 1023)/1024, 1024, 0, funcStream1>>>(0, numOfPixels, Y_1, U_1);

        // cudaSetDevice(1);
    }

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<chrono::microseconds>(stop - start).count();
    cout << "Net duration of visual pass : " << duration << endl;

    cudaSetDevice(0);
    // cout << (int) hostTest[2] << endl;
    cudaMemcpy(hostTest, rgcCurrents, numOfPixels*sizeof(float), cudaMemcpyDeviceToHost);
    // cout << hostTest[125100] << endl;
    for(int i = 0; i < 323350; i +=1){
        if(hostTest[i] != i){
            cout << "Unmatch at : " << i << " / "<< hostTest[i] << endl;
        }
    }



//            cudaMemcpyAsync(frame->data[2], V_2, numOfPixels*sizeof(float), cudaMemcpyDeviceToHost,stream1);
// cout << "TestVal : " << (int) y1[2] << endl;

    cudaFree(Y_1);
//    cudaFree(Y_1);
//    cudaFree(Y_2);


}

//int mrain(vector<vector<float>> &gcuArr)
//{
//
//    visualPass1();
//    int N = 1<<27;
//    int deviceCount;
//    cudaGetDeviceCount(&deviceCount);
//    for(int i = 0; i < 4; i+=1){
//        vector<float> temp(1<<27, 5);
//        gcuArr.push_back(temp);
//    }
//    gcuArr[0][401] = 7.99;
//    gcuArr[1][401] = 4;
//    gcuArr[2][401] = 8;
//    gcuArr[3][401] = 12.42;
//    float *d_1, *d_2, *d_3, *d_4;
//
//    cudaStream_t stream1, stream2, stream3, stream4 ;
//
//    cudaSetDevice(0);
//    cudaStreamCreate ( &stream1) ;
//    cudaStreamCreate ( &stream3) ;
//    cudaMalloc(&d_1, N*sizeof(float));
//    cudaMalloc(&d_2, N*sizeof(float));
//    cudaSetDevice(1);
//    cudaStreamCreate(&stream2);
//    cudaStreamCreate ( &stream4) ;
//    cudaMalloc(&d_3, N*sizeof(float));
//    cudaMalloc(&d_4, N*sizeof(float));
//
////    for (int i = 0; i < deviceCount; i ++){
////        cudaSetDevice(i);
////        cudaMalloc(&d_x, N*sizeof(float));
////    }
//
//
////    cudaStreamCreate(&stream2);
////    cudaStreamCreate ( &stream3) ;
////    cudaStreamCreate ( &stream4) ;
//
//
//    cudaSetDevice(0);
//    cudaMemcpyAsync(d_1, gcuArr[0].data(), N*sizeof(float), cudaMemcpyHostToDevice, stream1);
//    cudaMemcpyAsync(d_2, gcuArr[1].data(), N*sizeof(float), cudaMemcpyHostToDevice, stream1);
//    cudaDeviceSynchronize();
//    cudaSetDevice(1);
//    cudaMemcpyAsync(d_3, gcuArr[2].data(), N*sizeof(float), cudaMemcpyHostToDevice, stream2);
//    cudaMemcpyAsync(d_4, gcuArr[3].data(), N*sizeof(float), cudaMemcpyHostToDevice, stream2);
//    cudaDeviceSynchronize();
//
//
//    auto start = std::chrono::high_resolution_clock::now();
//
//    for (int i = 0; i < 1000; i+=1){
//        cudaSetDevice(0);
//        saxpy<<<(N + 1023)/1024, 1024, 1024 * sizeof(double), stream1>>>(N, 2.0f, d_1, d_2);
//        cudaSetDevice(1);
//        saxpy<<<(N + 1023)/1024, 1024, 1024 * sizeof(double), stream2>>>(N, 2.0f, d_3, d_4);
//        // cudaDeviceSynchronize();
//    }
//    cudaSetDevice(0);
//    cudaDeviceSynchronize();
//    cudaSetDevice(1);
//    cudaDeviceSynchronize();
//
//    auto stop = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<chrono::microseconds>(stop - start).count();
//    std::cout << "Time taken :         " << duration << endl;
//
//
//    cudaSetDevice(0);
//    cudaMemcpyAsync(gcuArr[0].data(), d_1, N*sizeof(float), cudaMemcpyDeviceToHost, stream1);
//    cudaMemcpyAsync(gcuArr[1].data(), d_2, N*sizeof(float), cudaMemcpyDeviceToHost,stream1);
//    cudaDeviceSynchronize();
//    cudaSetDevice(1);
//    cudaMemcpyAsync(gcuArr[2].data(), d_3, N*sizeof(float), cudaMemcpyDeviceToHost, stream2);
//    cudaMemcpyAsync(gcuArr[3].data(), d_4, N*sizeof(float), cudaMemcpyDeviceToHost, stream2);
//    cudaDeviceSynchronize();
//
//    cudaDeviceProp a{};
//
//    cudaGetDeviceProperties(&a, 0);
//    cudaSetDevice(1);
//    std::cout << "Number of devices :         " << deviceCount << endl;
//    std::cout << "Device name :               " << a.name << endl;
//    std::cout << "Test var 3 :                " << gcuArr[2][401] << endl;
//
//    float maxError = 0.0f;
////    for (int i = 0; i < N; i++)
////        maxError = max(maxError, abs(y[i]-4.0f));
//    printf("Max error: %f\n", maxError);
//
//    cudaFree(d_1);
//    cudaFree(d_2);
//    cudaFree(d_3);
//    cudaFree(d_4);
//}
//
//int testFunct (){
//    cudaProfilerStart();
//    int N = 1 << 27;
//    int deviceCount;
//    cudaGetDeviceCount(&deviceCount);
//    cout << N << endl;
//
//
//
//    float *aDest = (float*)malloc(N*sizeof(float));
//    float *bDest = (float*)malloc(N*sizeof(float));
//    float *cDest = (float*)malloc(N*sizeof(float));
//    float *dDest = (float*)malloc(N*sizeof(float));
//    float *aHost = (float*)malloc(N*sizeof(float));
//    float *bHost = (float*)malloc(N*sizeof(float));
//    float *cHost = (float*)malloc(N*sizeof(float));
//    float *dHost = (float*)malloc(N*sizeof(float));
////    vector<float[] > trial, hostMem;
////    trial.push_back(aDest); trial.push_back(bDest);
////    hostMem.push_back(aHost); hostMem.push_back(bHost);
//    vector<cudaStream_t> memStream, functStream;
//
//    cudaMallocHost((void**)&aHost, N*sizeof(float));
//    cudaMallocHost((void**)&bHost, N*sizeof(float));
//    cudaMallocHost((void**)&cHost, N*sizeof(float));
//    cudaMallocHost((void**)&dHost, N*sizeof(float));
//    for(int i = 0; i < N; i+=1){
//        aDest[i] =  i;
//        bDest[i] =  (i - 2);
//        cDest[i] =  i;
//        dDest[i] =  (i - 2);
//        aHost[i] =  2;
//        bHost[i] =  67.83;
//        cHost[i] =  2;
//        dHost[i] =  67.83;
//    }
//
//
//    cudaStream_t stream1, stream2, stream3, stream4;
//    for (int i = 0; i < deviceCount; i+=1){
//        cudaStream_t temp;
//        memStream.push_back(temp);
//        cudaStream_t temp1;
//        functStream.push_back(temp1);
//        cudaSetDevice(i);
//        cudaStreamCreate(&memStream[i]);
//        cudaStreamCreate(&functStream[i]);
//    }
//    cudaSetDevice(0);
//    cudaStreamCreate(&stream1);
//    cudaStreamCreate(&stream3);
//    cudaMalloc(&aDest, N * sizeof(float));
//    cudaMalloc(&bDest, N * sizeof(float));
//    cudaSetDevice(1);
//    cudaStreamCreate(&stream2);
//    cudaStreamCreate(&stream4);
//    cudaMalloc(&cDest, N * sizeof(float));
//    cudaMalloc(&dDest, N * sizeof(float));
//
//
//    auto start = std::chrono::high_resolution_clock::now();
////    std::vector<std::thread> threads;
////
////    for (unsigned int device_id = 0; device_id < deviceCount; device_id++)
////    {
////        threads.push_back (std::thread ([&,device_id] () {
////            cudaSetDevice (device_id);
////            if(device_id == 0){
////                for(int i = 0; i < 1000; i +=1){
////                    if((i % 36 == 1) || (i == 1)) {
////              cudaMemcpyAsync(bDest, bHost, N* sizeof(float), cudaMemcpyHostToDevice, memStream[0]);
////              cudaMemcpyAsync(bDest, aHost, N* sizeof(float), cudaMemcpyHostToDevice, memStream[0]);
////                    }
////              eye1Pipeline<<<(N+1023)/1024, 1024, 0, functStream[0]>>>(0, 0, aDest, bDest);
////                    // cudaStreamSynchronize(stream3);
////                }
////
////                // cudaDeviceSynchronize();
////                // cudaMemcpyAsync(frameStorage[4], oneGuy, numOfPixels * sizeof(float), cudaMemcpyDeviceToHost, stream1);
////                // cudaStreamSynchronize(stream1);
////            } else if(device_id == 1){
////                for(int i = 0; i < 1000; i +=1){
////                    if((i % 36 == 1) || (i == 1)) {
////                cudaMemcpyAsync(cDest, aHost, N* sizeof(float), cudaMemcpyHostToDevice, memStream[1]);
////                //cudaMemcpyAsync(dDest, bHost, N* sizeof(float), cudaMemcpyHostToDevice, memStream[1]);
////                    }
////                    // eye1Pipeline<<<(numOfPixels + 1023)/1024, 1024, 0, stream4>>>(0, numOfPixels, Y_2,U_2);
////                    // cudaStreamSynchronize(stream4);
////                }
////            }
////        }));
////    }
////    for (auto &thread: threads)
////        thread.join ();
//
//    for(int i = 0; i < 2; i++){
//        cudaSetDevice(0);
//        cudaMemcpyAsync(aDest, aHost, N* sizeof(float), cudaMemcpyHostToDevice, stream1);
//        cudaMemcpyAsync(bDest, aHost, N* sizeof(float), cudaMemcpyHostToDevice, stream3);
//        cudaSetDevice(1);
//        cudaMemcpyAsync(cDest, aHost, N* sizeof(float), cudaMemcpyHostToDevice,stream2);
//        // cudaMemcpyAsync(dDest, dHost, N* sizeof(float), cudaMemcpyHostToDevice, stream4);
//
////        cudaStreamSynchronize(stream1);
////        cudaStreamSynchronize(stream2);
////        cudaStreamSynchronize(stream3);
////        cudaStreamSynchronize(stream4);
////        cudaSetDevice(0);
////        eye1Pipeline<<<(N+1023)/1024, 1024, 0, functStream[0]>>>(0, 0, aDest, bDest);
////        cudaSetDevice(1);
////        eye1Pipeline<<<(N+1023)/1024, 1024, 0, functStream[1]>>>(0, 0, cDest, dDest);
//
//    }
////#pragma omp parallel for num_threads(2)
////    for(int i = 0; i < 2; i++){
////        cudaSetDevice(i);
////        if(i == 0) {
////            for(int j = 0; j < 2; j++){
////                cudaMemcpyAsync(aDest, aHost, N* sizeof(float), cudaMemcpyHostToDevice, stream1);
////                cudaMemcpyAsync(bDest, bHost, N* sizeof(float), cudaMemcpyHostToDevice, stream3);
////            }
////        } else if(i == 1){
////            for(int j = 0; j < 2; j++){
////                cudaMemcpyAsync(cDest, cHost, N* sizeof(float), cudaMemcpyHostToDevice,stream2);
////                // cudaMemcpyAsync(dDest, dHost, N* sizeof(float), cudaMemcpyHostToDevice, stream4);
////            }
////        }
////
////    }
//
////    cudaStreamSynchronize(stream1);
////    cudaStreamSynchronize(stream2);
////    cudaStreamSynchronize(stream3);
////    cudaStreamSynchronize(stream4);
//
//
//    auto stop = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<chrono::microseconds>(stop - start).count();
//    cout << "Net duration of visual pass : " << duration << endl;
//
////    cout << aHost[2] << endl;
////    cudaSetDevice(0);
////    cudaMemcpyAsync(aHost, aDest, N* sizeof(float), cudaMemcpyDeviceToHost, stream1);
////    cudaMemcpyAsync(bHost, bDest, N* sizeof(float), cudaMemcpyDeviceToHost, stream1);
////    cudaStreamSynchronize(stream1);
////
////    cout << aHost[2] << endl;
//
////    cudaFreeHost(aHost);
////    cudaFreeHost(bHost);
////    cudaFreeHost(cHost);
////    cudaFreeHost(dHost);
//    cudaFree(aDest);
//    cudaFree(bDest);
//    cudaProfilerStop();
//    cudaDeviceReset();
//}

// ---------------- Rough Pad ------------------------

//    for (unsigned int device_id = 0; device_id < deviceCount; device_id++)
//    {
//        threads.push_back (std::thread ([&,device_id] () {
//            cudaSetDevice (device_id);
//            if(device_id == 0){
//                for(int i = 0; i < 1000; i +=1){
//                    if((i % 36 == 1) || (i == 1)) {
//                        cudaMemcpyAsync(Y_1, frameStorage[0], numOfPixels * sizeof(::uint8_t), cudaMemcpyHostToDevice,
//                                        stream1);
//                        cudaMemcpyAsync(U_1, frameStorage[1], numOfPixels * sizeof(::uint8_t), cudaMemcpyHostToDevice,
//                                        stream1);
//                        //cudaStreamSynchronize(stream1);
//                    }
//                    // eye1Pipeline<<<(numOfPixels + 1023)/1024, 1024, 0, stream3>>>(0, numOfPixels, oneGuy);
//                    // cudaStreamSynchronize(stream3);
//                }
//
//                // cudaDeviceSynchronize();
//                // cudaMemcpyAsync(frameStorage[4], oneGuy, numOfPixels * sizeof(float), cudaMemcpyDeviceToHost, stream1);
//                // cudaStreamSynchronize(stream1);
//            } else if(device_id == 1){
//                for(int i = 0; i < 1000; i +=1){
//                    if((i % 36 == 1) || (i == 1)) {
//                        cudaMemcpyAsync(Y_2, frameStorage[2], numOfPixels * sizeof(::uint8_t), cudaMemcpyHostToDevice,
//                                        stream2);
//                        cudaMemcpyAsync(U_2, frameStorage[3], numOfPixels * sizeof(::uint8_t), cudaMemcpyHostToDevice,
//                                        stream2);
//                        //cudaStreamSynchronize(stream2);
//                    }
//                    // eye1Pipeline<<<(numOfPixels + 1023)/1024, 1024, 0, stream4>>>(0, numOfPixels, Y_2,U_2);
//                    // cudaStreamSynchronize(stream4);
//                }
//            }
//        }));
//    }
//    for (auto &thread: threads)
//        thread.join ();
//    for (int milliCount = 0; milliCount < 100; milliCount+=1){
////        int64_t pts;
////        AVFrame* frame;
//
//        //if((milliCount % 36 == 1) || (milliCount == 1)){
//            //           frame = video_reader_read_frame(&vr_state, frame_data, &pts);
//            // ::memcpy(pinnedTemp, frameStorage[frameIndex], 6); frameIndex+=1;
//            cudaSetDevice(0);
//            cudaMemcpyAsync(Y_1, frameStorage[0], numOfPixels*sizeof(::uint8_t), cudaMemcpyHostToDevice, stream1);
//            cudaMemcpyAsync(U_1, frameStorage[1], numOfPixels*sizeof(::uint8_t), cudaMemcpyHostToDevice, stream1);
////            cudaMemcpyAsync(V_1, frame->data[2], numOfPixels*sizeof(float), cudaMemcpyHostToDevice, stream1);
//            // cudaDeviceSynchronize();
//            cudaSetDevice(1);
//             cudaMemcpyAsync(Y_2, frameStorage[2], numOfPixels*sizeof(::uint8_t), cudaMemcpyHostToDevice, stream2);
//            cudaMemcpyAsync(U_2, frameStorage[3], numOfPixels*sizeof(::uint8_t), cudaMemcpyHostToDevice, stream2);
////            cudaMemcpyAsync(V_2, frame->data[2], numOfPixels*sizeof(float), cudaMemcpyHostToDevice, stream2);
//            // cudaDeviceSynchronize();
//        //}
////        if(milliCount % 36 == 35){
////            cudaSetDevice(0);
////            cudaStreamSynchronize(stream1);
////            cudaSetDevice(1);
////            cudaStreamSynchronize(stream2);
////        }
//
////        cudaSetDevice(0);
////        eye1Pipeline<<<(numOfPixels + 1023)/1024, 1024, 0, stream3>>>(0, numOfPixels, nullptr, nullptr);
////        cudaStreamSynchronize(stream3);
////        cudaSetDevice(1);
////        eye1Pipeline<<<(numOfPixels + 1023)/1024, 1024, 0, stream4>>>(0, numOfPixels, nullptr, nullptr);
////        cudaStreamSynchronize(stream4);
//
////        if((milliCount % 36 == 1) || (milliCount == 1)){
////            cudaSetDevice(0);
////            cudaMemcpyAsync(frame->data[0], Y_1, numOfPixels*sizeof(float), cudaMemcpyDeviceToHost, stream1);
////            cudaMemcpyAsync(frame->data[1], U_1, numOfPixels*sizeof(float), cudaMemcpyDeviceToHost,stream1);
//////            cudaMemcpyAsync(frame->data[2], V_1, numOfPixels*sizeof(float), cudaMemcpyDeviceToHost,stream1);
////            // cudaStreamSynchronize(stream1);
////            cudaSetDevice(1);
////            cudaMemcpyAsync(frame->data[0], Y_2, numOfPixels*sizeof(float), cudaMemcpyDeviceToHost, stream2);
////            cudaMemcpyAsync(frame->data[1], U_2, numOfPixels*sizeof(float), cudaMemcpyDeviceToHost,stream2);
//////            cudaMemcpyAsync(frame->data[2], V_2, numOfPixels*sizeof(float), cudaMemcpyDeviceToHost,stream1);
////            // cudaStreamSynchronize(stream2);
////        }
//
//    }
