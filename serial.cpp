#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

// 1D-IDCT
vector<double> idct_1d(const vector<double>& signal) {
    int N = signal.size();
    vector<double> result(N, 0.0);
    for (int x = 0; x < N; ++x) {
        double sum_value = 0.0;
        for (int u = 0; u < N; ++u) {
            double alpha_u = (u == 0) ? 1.0 / sqrt(N) : sqrt(2.0 / N);
            sum_value += alpha_u * signal[u] * cos((CV_PI * (2 * x + 1) * u) / (2 * N));
        }
        result[x] = sum_value;
    }
    return result;
}

// 2D-IDCT using two 1D-IDCTs
Mat idct_2d(const Mat& dct_matrix) {
    int N = dct_matrix.rows;
    int M = dct_matrix.cols;
    Mat image = Mat::zeros(N, M, CV_32F);

    // Step 1: Apply 1D-IDCT to each column
    for (int j = 0; j < M; ++j) {
        vector<double> column(N);
        for (int i = 0; i < N; ++i) {
            column[i] = dct_matrix.at<float>(i, j);
        }
        vector<double> idct_column = idct_1d(column);
        for (int i = 0; i < N; ++i) {
            image.at<float>(i, j) = idct_column[i];
        }
    }

    // Step 2: Apply 1D-IDCT to each row
    for (int i = 0; i < N; ++i) {
        vector<double> row(M);
        for (int j = 0; j < M; ++j) {
            row[j] = image.at<float>(i, j);
        }
        vector<double> idct_row = idct_1d(row);
        for (int j = 0; j < M; ++j) {
            image.at<float>(i, j) = idct_row[j];
        }
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
            sum_value += signal[x] * cos((CV_PI * (2 * x + 1) * u) / (2 * N));
        }
        result[u] = (u == 0) ? sum_value * (1.0 / sqrt(N)) : sum_value * sqrt(2.0 / N);
    }
    return result;
}

// 2D-DCT using two 1D-DCTs
Mat dct_2d(const Mat& image) {
    int N = image.rows;
    int M = image.cols;
    Mat dct_matrix = Mat::zeros(N, M, CV_32F);

    // Step 1: Apply 1D-DCT to each row
    for (int i = 0; i < N; ++i) {
        vector<double> row(M);
        for (int j = 0; j < M; ++j) {
            row[j] = image.at<float>(i, j);
        }
        vector<double> dct_row = dct_1d(row);
        for (int j = 0; j < M; ++j) {
            dct_matrix.at<float>(i, j) = dct_row[j];
        }
    }

    // Step 2: Apply 1D-DCT to each column
    for (int j = 0; j < M; ++j) {
        vector<double> column(N);
        for (int i = 0; i < N; ++i) {
            column[i] = dct_matrix.at<float>(i, j);
        }
        vector<double> dct_column = dct_1d(column);
        for (int i = 0; i < N; ++i) {
            dct_matrix.at<float>(i, j) = dct_column[i];
        }
    }

    return dct_matrix;
}

// Calculate PSNR
double calculate_psnr(const Mat& original, const Mat& reconstructed) {
    Mat diff;
    absdiff(original, reconstructed, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    Scalar s = sum(diff);
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) { // for small values return zero
        return 100;
    } else {
        double mse = sse / (double)(original.total() * original.channels());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

int main() {
    // Load the image and convert to grayscale
    Mat image = imread("lena.png");
    Mat grayscale_image;
    cvtColor(image, grayscale_image, COLOR_BGR2GRAY);

    // Resize the image to 256x256
    resize(grayscale_image, grayscale_image, Size(256, 256));

    // Convert to float32 for OpenCV DCT
    Mat grayscale_image_float;
    grayscale_image.convertTo(grayscale_image_float, CV_32F);

    // Apply optimized 2D-DCT using two 1D-DCTs
    // auto start_time = chrono::high_resolution_clock::now();
    Mat dct_optimized = dct_2d(grayscale_image_float);
    // auto end_time = chrono::high_resolution_clock::now();
    // chrono::duration<double> elapsed = end_time - start_time;
    // cout << "Optimized 2D-DCT time (two 1D-DCTs): " << elapsed.count() << " seconds" << endl;

    Mat reconstructed_image = idct_2d(dct_optimized);
    // Clip to valid range [0, 255]
    reconstructed_image = min(max(reconstructed_image, 0), 255);
    reconstructed_image.convertTo(reconstructed_image, CV_8U);

    // Calculate PSNR between the original and reconstructed image
    double psnr_brute = calculate_psnr(grayscale_image, reconstructed_image);
    cout << "PSNR: " << psnr_brute << " dB" << endl;

    // Display results
    imshow("Original Image", grayscale_image);
    imshow("Reconstructed Image", reconstructed_image);
    waitKey(0);

    // Compare the results between brute force and optimized
    Mat log_dct_optimized;
    log(abs(dct_optimized) + 1, log_dct_optimized);
    imshow("DCT (Optimized using two 1D-DCT)", log_dct_optimized);
    waitKey(0);

    // start_time = chrono::high_resolution_clock::now();
    Mat dct_cv2;
    dct(grayscale_image_float, dct_cv2);
    // end_time = chrono::high_resolution_clock::now();
    // elapsed = end_time - start_time;
    // cout << "cv2 dct " << elapsed.count() << " seconds" << endl;

    Mat idct_cv2;
    idct(dct_cv2, idct_cv2);
    idct_cv2 = min(max(idct_cv2, 0), 255);
    idct_cv2.convertTo(idct_cv2, CV_8U);
    double psnr_cv2 = calculate_psnr(grayscale_image, idct_cv2);
    cout << "PSNR (cv2): " << psnr_cv2 << " dB" << endl;

    return 0;
}