#include "BlockMatcherGPU.h"

#include <iostream>
#include <string>
#include <vector>
#include <gdal.h>
#include <gdal_priv.h>

int main() {

    GDALAllRegister(); // Register GDAL drivers

    // Open the left and right images using GDAL
    GDALDataset *leftImageDataset = static_cast<GDALDataset *>(GDALOpen("/gpfsm/dnb06/projects/p206/code/hackathon/test_cuda_function/aster-L.tif", GA_ReadOnly));
    GDALDataset *rightImageDataset = static_cast<GDALDataset *>(GDALOpen("/gpfsm/dnb06/projects/p206/code/hackathon/test_cuda_function/aster-R.tif", GA_ReadOnly));

    if (!leftImageDataset || !rightImageDataset)
    {
        std::cerr << "Error: Unable to open input images using GDAL." << std::endl;
        GDALClose(leftImageDataset);
        GDALClose(rightImageDataset);
        return 1;
    }

    int cols = leftImageDataset->GetRasterXSize();
    int rows = leftImageDataset->GetRasterYSize();

    std::cout << "Got cols and rows\n" << " ";
    /*
    std::vector<std::vector<double>> left_image(rows, std::vector<double>(cols));
    std::vector<std::vector<double>> right_image(rows, std::vector<double>(cols));

    leftImageDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, cols, rows, left_image.data(), cols, rows, GDT_Float32, 0, 0);
    rightImageDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, cols, rows, right_image.data(), cols, rows, GDT_Float32, 0, 0);
    */

    // Create C-style arrays for left and right images
    double **left_image = new double *[rows];
    double **right_image = new double *[rows];

    std::cout << "Allocating\n" << " ";

    for (int i = 0; i < rows; ++i)
    {
        left_image[i] = new double[cols];
        right_image[i] = new double[cols];
    }

    std::cout << "Reading in\n" << " ";
    // Read data into the C-style arrays
    leftImageDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, cols, rows, left_image, cols, rows, GDT_Float32, 0, 0);
    rightImageDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, cols, rows, right_image, cols, rows, GDT_Float32, 0, 0);
    std::cout << "Done copying\n" << " ";
    GDALClose(leftImageDataset);
    GDALClose(rightImageDataset);
    std::cout << "Closed\n" << " ";

    /*
    BlockMatcherGPU blockmatching(rows, cols, 21, 20);

    // Compute the disparity map
    blockmatching.compute_disparity(left_image, right_image);

    // Get the disparity map
    std::vector<std::vector<double>> &disparity_map = blockmatching.getDisparityMap();

    // Create a GDAL dataset for the output disparity map
    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset *disparityDataset = driver->Create("disparity.tif", cols, rows, 1, GDT_Float64, nullptr);

    // Write disparity map data to the GDAL dataset
    disparityDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, cols, rows, disparity_map.data(), cols, rows, GDT_Float32, 0, 0);

    // Close the GDAL dataset
    GDALClose(disparityDataset);
    */

    return 0;
}