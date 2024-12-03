#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

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
