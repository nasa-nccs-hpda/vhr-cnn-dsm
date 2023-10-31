#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include "timer.h"
timer t0;
​
#define NVTX_START(name) nvtxRangePushA(name)
#define NVTX_STOP() nvtxRangePop()
​
using namespace std;
​
__device__ int f(int r, int c)
{
    return r^c;
}
​
__global__ void set2d(int *p, int width, int height) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int xthreads = gridDim.x * blockDim.x;
    int ythreads = gridDim.y * blockDim.y;
​
    for (int r = row; r < height; r += ythreads) {
        for (int c = col; c < width; c += xthreads) {
            p[r * width + c] = f(r, c);
        }
    }
}
​
void get_options(int argc, char *argv[], dim3 *threads, dim3 *blocks, int *m, int *n, int *fill_array)
{
    int c;
    while ((c = getopt(argc, argv, "m:n:v:z")) >= 0) {
        switch (c) {
            case 'm': *m = atoi(optarg); break;
            case 'n': *n = atoi(optarg); break;
            case 'v': sscanf(optarg, "%u,%u,%u,%u", &blocks->x, &blocks->y, &threads->x, &threads->y); break;
            case 'z': *fill_array = 0; break;
            default:
                      fprintf(stderr, "Usage: twodv [options]\n"
                              "-m X   -- Number of rows in array\n"
                              "-n X   -- Number of columns in array\n"
                              "-v X,Y,x,y -- Number of blocks (X,Y) and number of threads (x,y) to use\n"
                              "-z     -- Don't initialize the memory on the CPU\n"
                             );
                      exit(-1);
        }
    }
}
​
​
int main(int argc, char *argv[])
{
    int m = 1<<16;
    int n = 1<<10;
    dim3 threads = {32, 32};
    dim3 blocks;
    int fill_array = 1;
​
​
    NVTX_START("Initializing Cuda");
    cudaFree(0);
    int deviceID;
    cudaDeviceProp prop;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop, deviceID); 
    blocks = {(unsigned)prop.multiProcessorCount, 2};
    NVTX_STOP();
​
    get_options(argc, argv, &threads, &blocks, &m, &n, &fill_array);
​
    NVTX_START("Getting vector");
    vector<int> z;
    vector<vector<int> > a;
    a.resize(m);
    NVTX_STOP();
​
    NVTX_START("Filling vector");
    for (int i = 0; i < m; i++) {
        a[i].resize(n);
        if (fill_array) {
            for (int j = 0; j < n; j++) {
                a[i][j] = -7;
            }
        }
    }
    NVTX_STOP();
​
    NVTX_START("Allocating and copying GPU memory");
    int *p;
    cudaMallocManaged(&p, n*m*sizeof(int));
    if (fill_array) {
        for (int i = 0; i < m; i++) {
            int *pa = a[i].data();
            for (int j = 0; j < n; j++) {
                p[i*n+j] = pa[j];
            }
        }
    }
    NVTX_STOP();
​
    NVTX_START("GPU Exec");
    t0.tick();
    set2d<<<blocks, threads>>>(p, n, m);
    cudaDeviceSynchronize();
    double dt = t0.tock();
    NVTX_STOP();
​
    NVTX_START("D2H");
    int errors = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = p[i*n+j];
            if (a[i][j] < 0) {
                printf("Unset at (%d,%d) -> %d\n", i, j, a[i][j]);
                if (errors++ > 10) return -1;
            }
        }
    }
    NVTX_STOP();
​
    NVTX_START("Evaluating results");
    printf("%d-%d\n", 0, 16);
    for (int i = 0; i < 16; i++) {
        printf("%4d: ", i);
        for (int j = 0; j < 16; j++) {
            printf("%4d ", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");
​
    printf("%d-%d\n", n/2-16, n/2);
    for (int i = m/2-16; i < m/2; i++) {
        printf("%4d: ", i);
        for (int j = n/2-16; j < n/2; j++) {
            printf("%4d ", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");
​
    printf("%d-%d\n", n-16, n);
    for (int i = m-16; i < m; i++) {
        printf("%4d: ", i);
        for (int j = n-16; j < n; j++) {
            printf("%4d ", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");
​
    printf("Thread structure: (%u, %u) * (%u, %u) == %u threads\n", blocks.x, blocks.y, threads.x, threads.x, blocks.x*blocks.y*threads.x*threads.x);
    printf("Time: %f\n", dt);
    printf("GB: %f\n", m*n*sizeof(int)/1e9);
    printf("GB/s: %f\n", m*n*sizeof(int)/dt/1e9);
    NVTX_STOP();
​
    return 0;
}