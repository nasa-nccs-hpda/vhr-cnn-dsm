#include <iostream>
#include "BlockMatcherCPU.h" 
#include "BlockMatcherGPU.h" 
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <cassert>

int main() {
    int rows = 5700;
    int cols = 5300;
    int block_size = 21;
    int search_range = 20;

    // Create instances of the left and right image data (dummy data)
    std::vector<double> left_image(rows * cols, 0);
    std::vector<double> right_image(rows * cols, 0);

    // Populate the dummy data (e.g., random values for demonstration)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            left_image[i * cols + j] = static_cast<double>(rand() % 256);  // Random values between 0 and 255
            right_image[i * cols + j] = static_cast<double>(rand() % 256);
            //double s =  static_cast<double>(rand() % 256);
            //left_image[i * cols + j] = s;  // Random values between 0 and 255
            //right_image[i * cols + j] = s;
        }
    }

    // Create a BlockMatcherGPU instance
    BlockMatcherCPU blockMatchercpu(rows, cols, block_size, search_range);

    std::cout << "CPU rows: " << " ";
    std::cout << rows << " ";
    std::cout << cols << "\n";
    // Compute the disparity map using the dummy data
    blockMatchercpu.compute_disparity(left_image, right_image);

    // Access the disparity map (disparity values are stored in disparityProcessor.disparity_map)
    std::vector<double> disparity_map_cpu = blockMatchercpu.disparity_map;

    int idx0 = rand() % 999999;
    int idx1 = rand() % 999999;
    int idx2 = rand() % 999999;

    std::cout << disparity_map_cpu[idx0] << " DISPARITY_MAP_CPU " << " \n";
    std::cout << disparity_map_cpu[idx1] << " DISPARITY_MAP_CPU " << " \n";
    std::cout << disparity_map_cpu[idx2] << " DISPARITY_MAP_CPU " << " \n";

    std::cout << "GPU" << " ";

    BlockMatcherGPU blockMatchergpu(rows, cols, block_size, search_range);
    blockMatchergpu.compute_disparity(left_image, right_image);

    std::vector<double> disparity_map_gpu = blockMatchergpu.disparity_map;

    std::cout << disparity_map_gpu[idx0] << " DISPARITY_MAP_GPU " << " \n";
    std::cout << disparity_map_gpu[idx1] << " DISPARITY_MAP_GPU " << " \n";
    std::cout << disparity_map_gpu[idx2] << " DISPARITY_MAP_GPU " << " \n";

    for (int i = 0; i < disparity_map_cpu.size(); i++){
        // bool ass = disparity_map_cpu[i] == disparity_map_gpu[i];
        // std::cout << i << " " << ass << " " << disparity_map_cpu[i] << " " << disparity_map_gpu[i] << "\n";
        assert(disparity_map_cpu[i] == disparity_map_gpu[i]);
    }

    return 0;
}