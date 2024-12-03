#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

#include "../lib/CycleTimer.h"
#include "../lib/util.hpp"
#include "dct_cuda.h"

int main(int argc, char* argv[]) {
    cout << "Using CUDA for DCT and iDCT" << endl;

    // Image preprocessing
    vector<Mat> image_channels(3), compressed_channels(3);
    util::load_image("../data/original/lena.png", image_channels);

    // Run the CUDA DCT implementation
    double dct_startTime = CycleTimer::currentSeconds();
    for (int channel = 0; channel < 3; ++channel) {
        compressed_channels[channel] = dct_2d_cuda(image_channels[channel]);
    }
    double dct_endTime = CycleTimer::currentSeconds();

    // Save the DCT image
    Mat dct_image;
    merge(compressed_channels, dct_image);
    imwrite("output/dct_image_cuda.png", dct_image);

    // Reconstruct image
    vector<Mat> reconstructed_channels(3);
    double idct_startTime = CycleTimer::currentSeconds();
    for (int channel = 0; channel < 3; ++channel) {
        reconstructed_channels[channel] = idct_2d_cuda(compressed_channels[channel]);
    }
    double idct_endTime = CycleTimer::currentSeconds();

    // Clamp images
    Mat reconstructed_image;
    merge(reconstructed_channels, reconstructed_image);
    reconstructed_image = min(max(reconstructed_image, 0.0f), 255.0f);
    reconstructed_image.convertTo(reconstructed_image, CV_8U);
    imwrite("output/reconstructed_image_cuda.png", reconstructed_image);

    // Calculate PSNR
    double psnr_startTime = CycleTimer::currentSeconds();
    Mat image = imread("../data/original/lena.png", IMREAD_COLOR);
    // resize(image, image, Size(256, 256));
    double psnr = util::calculate_psnr(image, reconstructed_image);
    double psnr_endTime = CycleTimer::currentSeconds();

    // Print results
    printf("CUDA DCT time:\t\t\t[%.3f] s\n", (dct_endTime - dct_startTime));
    printf("CUDA iDCT time:\t\t\t[%.3f] s\n", (idct_endTime - idct_startTime));
    printf("PSNR time:\t\t\t[%.3f] s\n", (psnr_endTime - psnr_startTime));
    printf("PSNR:\t\t\t\t[%.3f] dB\n", psnr);
    return 0;
}
