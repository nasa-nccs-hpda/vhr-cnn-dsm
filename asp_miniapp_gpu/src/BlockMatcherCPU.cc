#include "BlockMatcherCPU.h"
#include <iostream>
#include <cmath>
#include "timer.h"
timer t1;

BlockMatcherCPU::BlockMatcherCPU(int rows, int cols, int block_size, int search_range) {
    r = rows;
    c = cols;
    this->block_size = block_size;
    half_block_size = block_size / 2;
    this->search_range = search_range;
    disparity_map.resize(rows * cols, 0.0);
}

void BlockMatcherCPU::compute_disparity(const std::vector<double>& left_image, const std::vector<double>& right_image) {
    t1.tick();
    int max_displacement = search_range;

    for (int i = half_block_size; i < r - half_block_size; i++) {
        for (int j = half_block_size; j < c - half_block_size; j++) {

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
    double dt = t1.tock();
    printf("[CPU] Time: %f\n", dt);
    printf("[CPU] GB: %f\n", r*c*sizeof(double)/1e9);
    printf("[CPU] GB/s: %f\n", r*c*sizeof(double)/dt/1e9);
}

std::vector<double>& BlockMatcherCPU::getDisparityMap() {
    return disparity_map;
}
