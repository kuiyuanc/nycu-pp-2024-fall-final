#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "lib/util.hpp"

using namespace std;
using namespace cv;

namespace dct_serial {

// 1D-IDCT
vector<double> idct_1d(const vector<double>& signal) {
    int N = signal.size();
    vector<double> result(N, 0.0);
    for (int x = 0; x < N; ++x) {
        double sum_value = 0.0;
        for (int u = 0; u < N; ++u) {
            sum_value += util::image::alpha_cache[u] * signal[u] * util::image::cos_cache[u][x];
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
            sum_value += signal[x] * util::image::cos_cache[u][x];
        }
        result[u] = sum_value * util::image::alpha_cache[u];
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
            const int block_rows = min(BLOCK_SIZE, rows - i);
            const int block_cols = min(BLOCK_SIZE, cols - j);

            // Step 1: Apply 1D-DCT to each row within the block
            for (int bi = 0; bi < block_rows; ++bi) {
                vector<double> row(block_cols, 0.0);
                for (int bj = 0; bj < block_cols; ++bj) {
                    row[bj] = image.at<float>(i + bi, j + bj);
                }
                vector<double> dct_row = dct_1d(row);
                for (int bj = 0; bj < block_cols; ++bj) {
                    dct_matrix.at<float>(i + bi, j + bj) = dct_row[bj];
                }
            }

            // Step 2: Apply 1D-DCT to each column within the block
            for (int bj = 0; bj < block_cols; ++bj) {
                vector<double> col(block_rows, 0.0);
                for (int bi = 0; bi < block_rows; ++bi) {
                    col[bi] = dct_matrix.at<float>(i + bi, j + bj);
                }
                vector<double> dct_col = dct_1d(col);
                for (int bi = 0; bi < block_rows; ++bi) {
                    dct_matrix.at<float>(i + bi, j + bj) = dct_col[bi];
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
            const int block_rows = min(BLOCK_SIZE, rows - i);
            const int block_cols = min(BLOCK_SIZE, cols - j);

            // Step 1: Apply 1D-IDCT to each column within the block
            for (int bj = 0; bj < block_cols; ++bj) {
                vector<double> col(block_rows, 0.0);
                for (int bi = 0; bi < block_rows; ++bi) {
                    col[bi] = dct_matrix.at<float>(i + bi, j + bj);
                }
                vector<double> idct_col = idct_1d(col);
                for (int bi = 0; bi < block_rows; ++bi) {
                    image.at<float>(i + bi, j + bj) = idct_col[bi];
                }
            }

            // Step 2: Apply 1D-IDCT to each row within the block
            for (int bi = 0; bi < block_rows; ++bi) {
                vector<double> row(block_cols, 0.0);
                for (int bj = 0; bj < block_cols; ++bj) {
                    row[bj] = image.at<float>(i + bi, j + bj);
                }
                vector<double> idct_row = idct_1d(row);
                for (int bj = 0; bj < block_cols; ++bj) {
                    image.at<float>(i + bi, j + bj) = idct_row[bj];
                }
            }
        }
    }

    return image;
}

void dct_3d(const util::image::Channel3d& original, util::image::Channel3d& dct) {
    for (int i{0}; i < 3; ++i) {
        dct[i] = dct_2d(original[i]);
    }
}

void idct_3d(const util::image::Channel3d& dct, util::image::Channel3d& reconstructed) {
    for (int i{0}; i < 3; ++i) {
        reconstructed[i] = idct_2d(dct[i]);
    }
}

void dct_4d(const vector<util::image::Channel3d>& originals, vector<util::image::Channel3d>& dcts, const int& num_threads_assigned = 1) {
    for (int i{0}; i < originals.size(); ++i) {
        dct_3d(originals[i], dcts[i]);
    }
}

void idct_4d(const vector<util::image::Channel3d>& dcts, vector<util::image::Channel3d>& reconstructeds, const int& num_threads_assigned = 1) {
    for (int i{0}; i < dcts.size(); ++i) {
        idct_3d(dcts[i], reconstructeds[i]);
    }
}

}  // namespace dct_serial
