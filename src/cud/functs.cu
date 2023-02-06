#include <cstdio>
#include <iostream>
#include <libavcodec/avcodec.h>
#include "functs.h"
#include <vector>
#include <chrono>

using namespace std;
__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    x[i] = x[i] + 23;
    y[i] = y[i] + 8;
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
    if (!video_reader_open(&vr_state, "/Users/vishnumohan/CLionProjects/Workin_Metal/assets/switz.mp4")) {
        cout << "ERROR!!" << endl;
        cout << "Couldn't open video file (make sure you set a video file that exists" << endl;
    }

    // Allocate frame buffer
    constexpr int ALIGNMENT = 128;
    const int frame_width = vr_state.width;
    const int frame_height = vr_state.height;
    uint8_t* frame_data;
    cout << frame_height << endl;
    printf("\x1B[34m                         \tWidth : \033[0m"); cout << frame_width << endl;
    if (posix_memalign((void**)&frame_data, ALIGNMENT, frame_width * frame_height * 4) != 0) {
        cout << "ERROR!!" << endl;
        printf("Couldn't allocate frame buffer\n");
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

}

int mrain(vector<vector<float>> &gcuArr)
{
    int N = 1<<27;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int i = 0; i < 4; i+=1){
        vector<float> temp(1<<27, 5);
        gcuArr.push_back(temp);
    }
    gcuArr[0][3] = 7.99;
    gcuArr[1][3] = 4;
    gcuArr[2][3] = 8;
    gcuArr[3][3] = 12.42;
    float *d_1, *d_2, *d_3, *d_4;

    cudaStream_t stream1, stream2, stream3, stream4 ;

    cudaSetDevice(0);
    cudaStreamCreate ( &stream1) ;
    cudaStreamCreate ( &stream3) ;
    cudaMalloc(&d_1, N*sizeof(float));
    cudaMalloc(&d_2, N*sizeof(float));
    cudaSetDevice(1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate ( &stream4) ;
    cudaMalloc(&d_3, N*sizeof(float));
    cudaMalloc(&d_4, N*sizeof(float));

//    for (int i = 0; i < deviceCount; i ++){
//        cudaSetDevice(i);
//        cudaMalloc(&d_x, N*sizeof(float));
//    }


//    cudaStreamCreate(&stream2);
//    cudaStreamCreate ( &stream3) ;
//    cudaStreamCreate ( &stream4) ;


    cudaSetDevice(0);
    cudaMemcpyAsync(d_1, gcuArr[0].data(), N*sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_2, gcuArr[1].data(), N*sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaMemcpyAsync(d_3, gcuArr[2].data(), N*sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_4, gcuArr[3].data(), N*sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaDeviceSynchronize();


    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i+=1){
        cudaSetDevice(0);
        saxpy<<<(N+1023)/1024, 1024, 0, stream1>>>(N, 2.0f, d_1, d_2);
        cudaSetDevice(1);
        saxpy<<<(N+1023)/1024, 1024, 0, stream2>>>(N, 2.0f, d_3, d_4);
        // cudaDeviceSynchronize();
    }
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<chrono::microseconds>(stop - start).count();
    std::cout << "Time taken :         " << duration << endl;


    cudaSetDevice(0);
    cudaMemcpyAsync(gcuArr[0].data(), d_1, N*sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(gcuArr[1].data(), d_2, N*sizeof(float), cudaMemcpyDeviceToHost,stream1);
    cudaDeviceSynchronize();
     cudaSetDevice(1);
    cudaMemcpyAsync(gcuArr[2].data(), d_3, N*sizeof(float), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(gcuArr[3].data(), d_4, N*sizeof(float), cudaMemcpyDeviceToHost, stream2);
    cudaDeviceSynchronize();

    cudaDeviceProp a{};

    cudaGetDeviceProperties(&a, 0);
    cudaSetDevice(1);
    std::cout << "Number of devices :         " << deviceCount << endl;
    std::cout << "Device name :               " << a.name << endl;
    std::cout << "Test var 3 :                " << gcuArr[1][3] << endl;

    float maxError = 0.0f;
//    for (int i = 0; i < N; i++)
//        maxError = max(maxError, abs(y[i]-4.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(d_1);
    cudaFree(d_2);
    cudaFree(d_3);
    cudaFree(d_4);
}
