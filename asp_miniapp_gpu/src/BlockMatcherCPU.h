#ifndef BLOCKMATCHERCPU_H
#define BLOCKMATCHERCPU_H

#include <vector>

class BlockMatcherCPU {
public:
    BlockMatcherCPU(int rows, int c, int block_size, int search_range);
    int r, c, block_size, half_block_size, search_range;
    std::vector<double> disparity_map;
    void compute_disparity(const std::vector<double>& left_image, const std::vector<double>& right_image);

    std::vector<double>& getDisparityMap();
private:
    double compute_box_sum(const std::vector<double>& kernelCutLeft,
                           const std::vector<double>& kernelCutRight);
    double compute_sos(const std::vector<double>& kernelCutLeft,
                           const std::vector<double>& kernelCutRight);
};

#endif // BlockMatcherCPU_H