//
// Created by kasm-user on 1/20/23.
//
#include <vector>

#ifndef TEST1_FUNCTS_H
#define TEST1_FUNCTS_H
typedef struct {
    double x,y,z;
} XYZ;

typedef struct {
    int foveaWidth, paraLength, periLength, oz1up, oz1side, oz2up, oz2side, oz3up, oz3side;
    int *divFactors;
} RGCPARAMS;

typedef struct {
    int r,g,b, a;
} RGB;

typedef struct {
    int axis;
    double value;
    double cvalue,svalue;
} TRANSFORM;

typedef struct {
    XYZ p1,p2,p3,p4;
} FRUSTUM;

typedef struct {
    int testr;
    int perspWidth;
    int perspHeight;
    int worldWidth;
    int worldHeight;
    double longmin,longmax;
    double latmin,latmax;
    double perspfov;         // FOV of perspective view, horizontal
    char outname[256];       // Will be created automatically if not set on command line, see -f
    int antialias;           // Supersampling antialiasing
    int antialias2;
    int circularcrop;        // Circular window
    double voffset;          // Vertical offaxis amount as percentage for shift lens
    double hoffset;          // Horizontal offaxis amount, eg: for zero parallax alignment for stereo
    int remap;               // Create remap filter files for ffmpeg
    int ntransform;
    TRANSFORM *transform;    // Rotation transformations, these are performed in (reverse) order as specified
    int debug;
} PARAMS;

#define XTILT 0
#define YROLL 1
#define ZPAN  2

#define ABS(x) (x < 0 ? -(x) : (x))
#define MIN(x,y) (x < y ? x : y)
#define MAX(x,y) (x > y ? x : y)
#define SIGN(x) (x < 0 ? (-1) : 1)
#define MODULUS(p) (sqrt(p.x*p.x + p.y*p.y + p.z*p.z))
void CalcFrustum(void);
XYZ CameraRay(double,double);
XYZ VectorSum(double,XYZ,double,XYZ,double,XYZ,double,XYZ);

int visualPass1();
int testFunct ();
int mrain (std::vector<std::vector<float>> &temp);
void saxpy();
void formRGCinputs();
void eye1Pipeline();
#endif //TEST1_FUNCTS_H
