#include "BlockMatcherGPU.h"
#include <iostream>
#include <cmath>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>
#include "timer.h"
timer t0;

#define NVTX_START(name) nvtxRangePushA(name)
#define NVTX_STOP() nvtxRangePop()

BlockMatcherGPU::BlockMatcherGPU(int rows, int cols, int block_size, int search_range) {
    r = rows;
    c = cols;
    this->block_size = block_size;
    half_block_size = block_size / 2;
    this->search_range = search_range;
    disparity_map.resize(rows * cols, 0.0);
}

double BlockMatcherGPU::compute_sos(const std::vector<double>& kernelCutLeft,
                                    const std::vector<double>& kernelCutRight) {
    if (kernelCutLeft.size() != kernelCutRight.size()) {
        return std::numeric_limits<double>::max();
    }

    double sum_of_squares = 0.0;

    // Iterate over the pixels in the cutouts and compute the Sum of Squares.
    for (size_t i = 0; i < kernelCutLeft.size(); i++) {
        double diff = kernelCutLeft[i] - kernelCutRight[i];
        sum_of_squares += diff * diff;
    }

    return sum_of_squares;
}

double BlockMatcherGPU::compute_box_sum(const std::vector<double>& kernelCutLeft,
                        const std::vector<double>& kernelCutRight) {
    double box_sum = 0.0;

    // Iterate over the pixels in the cutouts and compute the box sum
    for (size_t i = 0; i < kernelCutLeft.size(); i++) {
        double diff = kernelCutLeft[i] - kernelCutRight[i];
        box_sum += std::abs(diff); // Absolute difference for the box sum
    }

    return box_sum;
}

__global__ void compute_disparity_gpu(const double* left_image, const double* right_image,
                                      double* disparity_map, const int r, const int c,
                                      const int block_size, const int max_displacement,
                                      const int half_block_size, const int d_block_size) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int xthreads = gridDim.x * blockDim.x;
    int ythreads = gridDim.y * blockDim.y;

    for (int i = row + half_block_size; i < r - half_block_size; i += ythreads) {
        for (int j = col + half_block_size; j < c - half_block_size; j += xthreads) {

            int min_disparity = 0;
            double min_sos = 179769000000000000; // make it really large

            // Search within the specified search range
            for (int d = 0; d <= max_displacement; d++) {
                double sos = 0;

                for (int y = -half_block_size; y <= half_block_size; y++) {
                    for (int x = -half_block_size; x <= half_block_size; x++) {
                        int left_idx = (i + y) * c + (j + x);
                        int right_idx = (i + y) * c + (j + x + d);
                        double left_pixel = left_image[left_idx];
                        double right_pixel = right_image[right_idx];
                        double diff = left_pixel - right_pixel;
                        sos += diff * diff;
                    }
                }

                // Update the disparity if the SOS is smaller
                if (sos < min_sos) {
                    min_sos = sos;
                    min_disparity = d;
                }
            }

            // Store the disparity in the disparity map
            disparity_map[i * c + j] = min_disparity;
        }
    }
}

void BlockMatcherGPU::compute_disparity(const std::vector<double>& left_image, const std::vector<double>& right_image) {
    int max_displacement = search_range;
    // int block_size_1d = block_size * block_size;

    std::cout << "[GPU] BLOCKDEF" << "\n";
    dim3 threads = {32, 32};
    dim3 blocks;

    cudaFree(0);
    int deviceID;
    cudaDeviceProp prop;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop, deviceID); 
    blocks = {(unsigned)prop.multiProcessorCount, 2};

    // std::cout << "Num blocks: " << blocks << " \n"; 
    
    // dim3 blockDim(block_size, block_size);
    // dim3 gridDim((c - 2 * half_block_size + blockDim.x - 1) / blockDim.x + 1, (r - 2 * half_block_size + blockDim.y - 1) / blockDim.y + 1);

    std::cout << "[GPU] cudaMallocManaged step" << "\n";

    double *left_image_device;
    double *right_image_device;
    double *disparity_map_device;
    double *kernelCutLeft;
    double *kernelCutRight;
    cudaMallocManaged(&left_image_device, left_image.size()*sizeof(double));
    cudaMallocManaged(&right_image_device, right_image.size()*sizeof(double));
    cudaMallocManaged(&disparity_map_device, left_image.size()*sizeof(double));
    cudaMallocManaged(&kernelCutLeft, block_size*block_size*sizeof(double));
    cudaMallocManaged(&kernelCutRight, block_size*block_size*sizeof(double));

    std::cout << "[GPU] SIZE: " << left_image.size() << "\n";

    std::cout << "[GPU] ALLOCATING AND COPY" << "\n";
    NVTX_START("Allocating and copying GPU memory");
    for (int i = 0; i < left_image.size(); i++){
        left_image_device[i] = left_image[i];
        right_image_device[i] = right_image[i];
        disparity_map_device[i] = 0.0f;
    }
    NVTX_STOP();
    
    std::cout << "[GPU] STARTING COMP" << "\n";
    int d_block_size = block_size * block_size;
    
    NVTX_START("GPU Exec");
    t0.tick();
    compute_disparity_gpu<<<blocks, threads>>>(left_image_device, right_image_device, disparity_map_device, r, c, block_size, max_displacement, half_block_size, d_block_size);
    cudaDeviceSynchronize();
    double dt = t0.tock();
    NVTX_STOP();

    for (int i = 0; i < left_image.size(); i++){
        disparity_map[i] = disparity_map_device[i];
    }
    printf("[GPU] Thread structure: (%u, %u) * (%u, %u) == %u threads\n", blocks.x, blocks.y, threads.x, threads.x, blocks.x*blocks.y*threads.x*threads.x);
    printf("[GPU] Time: %f\n", dt);
    printf("[GPU] GB: %f\n", r*c*sizeof(double)/1e9);
    printf("[GPU] GB/s: %f\n", r*c*sizeof(double)/dt/1e9);

}

std::vector<double>& BlockMatcherGPU::getDisparityMap() {
    return disparity_map;
}
