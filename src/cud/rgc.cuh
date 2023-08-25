//
// Created by kasm-user on 7/26/23.
//

#ifndef TEST1_RGC_CUH
#define TEST1_RGC_CUH
typedef struct {
    int rgcArrayHeight, RGCcount, fps, *xWidths, *yMatches;
} vid_rgc_params;

void saxpy();
int rgcSpikers (queue<float**> *rihq_l, queue<float**> *rihq_r, vid2rgcParams *v2rp, queue<RGC**> *rgcdetails, mutex &rgcMut, condition_variable &rgcCond);

#endif //TEST1_RGC_CUH
