#ifndef STEREOCORRELATOR_H
#define STEREOCORRELATOR_H

#include <string>

class StereoCorrelator{
public:
    StereoCorrelator(const std::string& leftImagePath, const std::string& rightImagePath,
                    int block_size, int search_range, const std::string& outputImagePath);

    void calculateDisparityMap();

private:
    std::string leftImagePath;
    std::string rightImagePath;
    int block_size;
    int search_range;
    std::string outputImagePath;
};


#endif