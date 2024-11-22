#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;

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

// 2D-IDCT using two 1D-IDCTs
Mat idct_2d(const Mat& dct_matrix) {
    int rows = dct_matrix.rows;
    int cols = dct_matrix.cols;
    Mat image(rows, cols, CV_32F, Scalar(0));

    // Step 1: Apply 1D-IDCT to each column
    for (int j = 0; j < cols; ++j) {
        vector<double> col(rows);
        for (int i = 0; i < rows; ++i)
            col[i] = dct_matrix.at<float>(i, j);
        vector<double> idct_col = idct_1d(col);
        for (int i = 0; i < rows; ++i)
            image.at<float>(i, j) = idct_col[i];
    }

    // Step 2: Apply 1D-IDCT to each row
    for (int i = 0; i < rows; ++i) {
        vector<double> row(cols);
        for (int j = 0; j < cols; ++j)
            row[j] = image.at<float>(i, j);
        vector<double> idct_row = idct_1d(row);
        for (int j = 0; j < cols; ++j)
            image.at<float>(i, j) = idct_row[j];
    }

    return image;
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
        result[u] = sum_value * ((u == 0) ? 1 / sqrt(N) : sqrt(2.0 / N));
    }
    return result;
}

// 2D-DCT using two 1D-DCTs
Mat dct_2d(const Mat& image) {
    int rows = image.rows;
    int cols = image.cols;
    Mat dct_matrix(rows, cols, CV_32F);

    // Step 1: Apply 1D-DCT to each row
    for (int i = 0; i < rows; ++i) {
        vector<double> row(cols);
        for (int j = 0; j < cols; ++j)
            row[j] = image.at<float>(i, j);
        vector<double> dct_row = dct_1d(row);
        for (int j = 0; j < cols; ++j)
            dct_matrix.at<float>(i, j) = dct_row[j];
    }

    // Step 2: Apply 1D-DCT to each column
    for (int j = 0; j < cols; ++j) {
        vector<double> col(rows);
        for (int i = 0; i < rows; ++i)
            col[i] = dct_matrix.at<float>(i, j);
        vector<double> dct_col = dct_1d(col);
        for (int i = 0; i < rows; ++i)
            dct_matrix.at<float>(i, j) = dct_col[i];
    }

    return dct_matrix;
}

// PSNR Calculation
double calculate_psnr(const Mat& original, const Mat& reconstructed) {
    Mat diff;
    absdiff(original, reconstructed, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    double mse = sum(diff)[0] / (original.total());
    if (mse == 0) return 100;  // No error
    double max_pixel = 255.0;
    return 20.0 * log10(max_pixel / sqrt(mse));
}

int main() {
    // Load the image
    Mat image = imread("lena.png", IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not load image." << endl;
        return -1;
    }
    resize(image, image, Size(256, 256));
    Mat image_float;
    image.convertTo(image_float, CV_32F);

    // Record start time for entire process
    auto total_start = chrono::high_resolution_clock::now();

    // Apply optimized 2D-DCT using two 1D-DCTs
    vector<Mat> dct_channels(3);
    auto dct_start = chrono::high_resolution_clock::now();
    for (int channel = 0; channel < 3; ++channel) {
        Mat channel_data;
        extractChannel(image_float, channel_data, channel);
        dct_channels[channel] = dct_2d(channel_data);
    }
    auto dct_end = chrono::high_resolution_clock::now();
    cout << "Optimized 2D-DCT time: "
         << chrono::duration<double>(dct_end - dct_start).count() << " seconds" << endl;

    // Save the DCT image
    Mat dct_image;
    merge(dct_channels, dct_image);
    imwrite("dct_image.png", dct_image);

    // Apply optimized 2D-IDCT using two 1D-IDCTs
    auto idct_start = chrono::high_resolution_clock::now();
    vector<Mat> reconstructed_channels(3);
    for (int channel = 0; channel < 3; ++channel) {
        reconstructed_channels[channel] = idct_2d(dct_channels[channel]);
    }
    auto idct_end = chrono::high_resolution_clock::now();
    cout << "Optimized 2D-IDCT time: "
         << chrono::duration<double>(idct_end - idct_start).count() << " seconds" << endl;

    // Merge reconstructed channels and clip to valid range
    auto merge_start = chrono::high_resolution_clock::now();
    Mat reconstructed_image;
    merge(reconstructed_channels, reconstructed_image);
    reconstructed_image = min(max(reconstructed_image, 0.0f), 255.0f);
    reconstructed_image.convertTo(reconstructed_image, CV_8U);
    auto merge_end = chrono::high_resolution_clock::now();
    cout << "Merging and clipping time: "
         << chrono::duration<double>(merge_end - merge_start).count() << " seconds" << endl;

    // Save the reconstructed image
    imwrite("reconstructed_image.png", reconstructed_image);

    // Calculate PSNR
    auto psnr_start = chrono::high_resolution_clock::now();
    double psnr = calculate_psnr(image, reconstructed_image);
    auto psnr_end = chrono::high_resolution_clock::now();
    cout << "PSNR calculation time: "
         << chrono::duration<double>(psnr_end - psnr_start).count() << " seconds" << endl;
    cout << "PSNR: " << psnr << " dB" << endl;

    // Record end time for entire process
    auto total_end = chrono::high_resolution_clock::now();
    cout << "Total execution time: "
         << chrono::duration<double>(total_end - total_start).count() << " seconds" << endl;

    return 0;
}
