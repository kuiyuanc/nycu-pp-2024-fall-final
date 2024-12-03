#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
// #include <chrono>
#include "../lib/CycleTimer.h"

using namespace std;
using namespace cv;

#include "../lib/util.hpp"
#define BLOCK_SIZE 8

// 1D-IDCT
vector<double> idct_1d(const vector<double>& signal) {
    int N = signal.size();
    vector<double> result(N, 0.0);
    for (int x = 0; x < N; ++x) {
        double sum_value = 0.0;
        for (int u = 0; u < N; ++u) {
            double alpha_u = (u == 0) ? 1.0 / sqrt(N) : sqrt(2.0 / N);
            sum_value += alpha_u * signal[u] * cos(M_PI * (2 * x + 1) * u / (2 * N));
        }
        result[x] = sum_value;
    }
    return result;
}

// 1D-DCT
vector<double> dct_1d(const vector<double>& signal) {
    int N = signal.size();
    vector<double> result(N, 0.0);
    for (int u = 0; u < N; ++u) {
        double sum_value = 0.0;
        for (int x = 0; x < N; ++x) {
            sum_value += signal[x] * cos(M_PI * (2 * x + 1) * u / (2 * N));
        }
        result[u] = sum_value * ((u == 0) ? 1.0 / sqrt(N) : sqrt(2.0 / N));
    }
    return result;
}

// 2D-DCT using two 1D-DCTs
Mat dct_2d(const Mat& image) {
    int rows = image.rows;
    int cols = image.cols;
    Mat dct_matrix(rows, cols, CV_32F);

    // Process each 8x8 block
    for (int i = 0; i < rows; i += BLOCK_SIZE) {
        for (int j = 0; j < cols; j += BLOCK_SIZE) {
            // Step 1: Apply 1D-DCT to each row within the block
            for (int bi = 0; bi < BLOCK_SIZE; ++bi) {
                if (i + bi >= rows) break;  // Boundary check
                vector<double> row(BLOCK_SIZE, 0.0);
                for (int bj = 0; bj < BLOCK_SIZE; ++bj) {
                    if (j + bj < cols) {
                        row[bj] = image.at<float>(i + bi, j + bj);
                    }
                }
                vector<double> dct_row = dct_1d(row);
                for (int bj = 0; bj < BLOCK_SIZE; ++bj) {
                    if (j + bj < cols) {
                        dct_matrix.at<float>(i + bi, j + bj) = dct_row[bj];
                    }
                }
            }

            // Step 2: Apply 1D-DCT to each column within the block
            for (int bj = 0; bj < BLOCK_SIZE; ++bj) {
                if (j + bj >= cols) break;  // Boundary check
                vector<double> col(BLOCK_SIZE, 0.0);
                for (int bi = 0; bi < BLOCK_SIZE; ++bi) {
                    if (i + bi < rows) {
                        col[bi] = dct_matrix.at<float>(i + bi, j + bj);
                    }
                }
                vector<double> dct_col = dct_1d(col);
                for (int bi = 0; bi < BLOCK_SIZE; ++bi) {
                    if (i + bi < rows) {
                        dct_matrix.at<float>(i + bi, j + bj) = dct_col[bi];
                    }
                }
            }
        }
    }

    return dct_matrix;
}

// 2D-IDCT using 8x8 blocks
Mat idct_2d(const Mat& dct_matrix) {
    int rows = dct_matrix.rows;
    int cols = dct_matrix.cols;
    Mat image(rows, cols, CV_32F, Scalar(0));

    // Process each 8x8 block
    for (int i = 0; i < rows; i += BLOCK_SIZE) {
        for (int j = 0; j < cols; j += BLOCK_SIZE) {
            // Step 1: Apply 1D-IDCT to each column within the block
            for (int bj = 0; bj < BLOCK_SIZE; ++bj) {
                if (j + bj >= cols) break;  // Boundary check
                vector<double> col(BLOCK_SIZE, 0.0);
                for (int bi = 0; bi < BLOCK_SIZE; ++bi) {
                    if (i + bi < rows) {
                        col[bi] = dct_matrix.at<float>(i + bi, j + bj);
                    }
                }
                vector<double> idct_col = idct_1d(col);
                for (int bi = 0; bi < BLOCK_SIZE; ++bi) {
                    if (i + bi < rows) {
                        image.at<float>(i + bi, j + bj) = idct_col[bi];
                    }
                }
            }

            // Step 2: Apply 1D-IDCT to each row within the block
            for (int bi = 0; bi < BLOCK_SIZE; ++bi) {
                if (i + bi >= rows) break;  // Boundary check
                vector<double> row(BLOCK_SIZE, 0.0);
                for (int bj = 0; bj < BLOCK_SIZE; ++bj) {
                    if (j + bj < cols) {
                        row[bj] = image.at<float>(i + bi, j + bj);
                    }
                }
                vector<double> idct_row = idct_1d(row);
                for (int bj = 0; bj < BLOCK_SIZE; ++bj) {
                    if (j + bj < cols) {
                        image.at<float>(i + bi, j + bj) = idct_row[bj];
                    }
                }
            }
        }
    }

    return image;
}

int main() {
    // Load the image
    Mat image = imread("../data/original/lena.png", IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not load image." << endl;
        return -1;
    }
    // resize(image, image, Size(256, 256));
    Mat image_float;
    image.convertTo(image_float, CV_32F);

    // Record start time for entire process
    double total_start = CycleTimer::currentSeconds();

    // Apply optimized 2D-DCT using two 1D-DCTs
    vector<Mat> dct_channels(3);
    double dct_start = CycleTimer::currentSeconds();
    for (int channel = 0; channel < 3; ++channel) {
        Mat channel_data;
        extractChannel(image_float, channel_data, channel);
        dct_channels[channel] = dct_2d(channel_data);
    }
    double dct_end = CycleTimer::currentSeconds();
    printf("Optimized 2D-DCT time:\t\t[%.3f] seconds\n", dct_end - dct_start);

    // Save the DCT image
    Mat dct_image;
    merge(dct_channels, dct_image);
    imwrite("output/dct_image.png", dct_image);

    // Apply optimized 2D-IDCT using two 1D-IDCTs
    double idct_start = CycleTimer::currentSeconds();
    vector<Mat> reconstructed_channels(3);
    for (int channel = 0; channel < 3; ++channel) {
        reconstructed_channels[channel] = idct_2d(dct_channels[channel]);
    }
    double idct_end = CycleTimer::currentSeconds();
    printf("Optimized 2D-IDCT time:\t\t[%.3f] seconds\n", idct_end - idct_start);

    // Merge reconstructed channels and clip to valid range
    double merge_start = CycleTimer::currentSeconds();
    Mat reconstructed_image;
    merge(reconstructed_channels, reconstructed_image);
    reconstructed_image = min(max(reconstructed_image, 0.0f), 255.0f);
    reconstructed_image.convertTo(reconstructed_image, CV_8U);
    double merge_end = CycleTimer::currentSeconds();
    printf("Merging and clipping time:\t[%.3f] seconds\n", merge_end - merge_start);

    // Save the reconstructed image
    imwrite("output/reconstructed_image.png", reconstructed_image);

    // Calculate PSNR
    double psnr_start = CycleTimer::currentSeconds();
    double psnr = util::calculate_psnr(image, reconstructed_image);
    double psnr_end = CycleTimer::currentSeconds();
    printf("PSNR calculation time:\t\t[%.3f] seconds\n", psnr_end - psnr_start);
    printf("PSNR:\t\t\t\t[%.3f] dB\n", psnr);

    // Record end time for entire process
    double total_end = CycleTimer::currentSeconds();
    printf("Total execution time:\t\t[%.3f] seconds\n", total_end - total_start);

    return 0;
}
