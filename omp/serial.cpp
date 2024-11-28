#include <vector>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include "common/CycleTimer.h"

using namespace std;
using namespace cv;



#include "dct.h"

void load_image(string filename, vector<Mat>& image_channels) {
    // 載入圖片
    Mat image = imread("lena.png", IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not load image." << endl;
        // return -1;
    }
    resize(image, image, Size(256, 256));
    Mat image_float;
    image.convertTo(image_float, CV_32F);
    vector<Mat> channels(3);
    split(image_float, channels);
    image_channels = channels;
}

void compress_image_3d(vector<Mat>& image_channels, vector<Mat>& compressed_channels) {
#pragma omp parallel for schedule(dynamic)
    for (int channel = 0; channel < 3; ++channel) {
        compressed_channels[channel] = dct_2d(image_channels[channel]);
    }
}

void reconstruct_image(vector<Mat>& compressed_channels, vector<Mat>& reconstructed_channels) {
#pragma omp parallel for schedule(dynamic)
    for (int channel = 0; channel < 3; ++channel) {
        reconstructed_channels[channel] = idct_2d(compressed_channels[channel]);
    }
}


int main(int argc, char* argv[]) {

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_filename>" << endl;
        return -1;
    }

    // printf("%s", argv[1]);
    // const int num_threads = omp_get_max_threads();
    const int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    cout << "Using " << num_threads << " threads" << endl;

    //
    // Image preprocessing.
    //
    vector<Mat> image_channels(3), compressed_channels(3);
    load_image("lena.png", image_channels);


    //
    // Run the serial implementation.
    //

    double dct_startTime = CycleTimer::currentSeconds();
    compress_image_3d(image_channels, compressed_channels);
    double dct_endTime = CycleTimer::currentSeconds();


    //
    // Save the DCT image
    //
    Mat dct_image;
    merge(compressed_channels, dct_image);
    imwrite("output/dct_image.png", dct_image);


    //
    // Reconstruct image
    //
    vector<Mat> reconstructed_channels(3);
    double idct_startTime = CycleTimer::currentSeconds();
    reconstruct_image(compressed_channels, reconstructed_channels);
    double idct_endTime = CycleTimer::currentSeconds();




    //
    // Clamp images
    //
    Mat reconstructed_image;
    merge(reconstructed_channels, reconstructed_image);
    reconstructed_image = min(max(reconstructed_image, 0.0f), 255.0f);
    reconstructed_image.convertTo(reconstructed_image, CV_8U);
    imwrite("output/reconstructed_image.png", reconstructed_image);


    //
    // Calculate PSNR
    //
    double psnr_startTime = CycleTimer::currentSeconds();
    Mat image = imread("lena.png", IMREAD_COLOR);
    resize(image, image, Size(256, 256));
    double psnr = calculate_psnr(image, reconstructed_image);
    double psnr_endTime = CycleTimer::currentSeconds();



    //
    // Print results
    //

    printf("Optimized 2D-DCT time:\t\t[%.3f] s\n", (dct_endTime - dct_startTime));
    // cout << "Optimized 2D-DCT time: " << chrono::duration<double>(dct_end - dct_start).count() << " seconds" << endl;
    // cout << "Optimized 2D-IDCT time: " << chrono::duration<double>(idct_end - idct_start).count() << " seconds" << endl;
    // cout << "PSNR calculation time: " << chrono::duration<double>(psnr_end - psnr_start).count() << " seconds" << endl;
    printf("PSNR:\t\t\t\t[%.3f] dB\n", psnr);
    return 0;
}