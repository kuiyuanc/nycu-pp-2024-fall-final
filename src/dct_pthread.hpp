#pragma once

#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

#include <pthread.h>

#include "../lib/util.hpp"

using namespace std;
using namespace cv;

namespace dct_pthread {
// direction of 1D
namespace direction {
constexpr int kRow{0};
constexpr int kCol{1};
}  // namespace direction

// read-only
namespace ro {
namespace idct {
Mat dct_channel;
}  // namespace idct

namespace dct {
Mat channel_data;
}  // namespace dct

int    rows;
int    cols;
int    block_rows;
int    rows_per_thread;
int    block_rows_per_thread;
int    num_threads;
string partition;
}  // namespace ro

// read-write
namespace rw {
namespace idct {
Mat image;
}  // namespace idct

namespace dct {
Mat dct_matrix;
}  // namespace dct
}  // namespace rw

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

void* sub_idct_2d(void* thread) {
    long       thread_id{long(thread)};
    const Mat& dct_matrix{ro::idct::dct_channel};
    for (int i = thread_id * ro::rows_per_thread; i < (thread_id + 1) * ro::rows_per_thread; i += util::image::kBlockSize) {
        for (int j = 0; j < ro::cols; j += util::image::kBlockSize) {
            const int block_rows = min(util::image::kBlockSize, ro::rows - i);
            const int block_cols = min(util::image::kBlockSize, ro::cols - j);

            // Step 1: Apply 1D-IDCT to each column within the block
            for (int bj = 0; bj < block_cols; ++bj) {
                vector<double> col(block_rows);
                for (int bi = 0; bi < block_rows; ++bi)
                    col[bi] = dct_matrix.at<float>(i + bi, j + bj);
                vector<double> idct_col = idct_1d(col);
                for (int bi = 0; bi < block_rows; ++bi)
                    rw::idct::image.at<float>(i + bi, j + bj) = idct_col[bi];
            }

            // Step 2: Apply 1D-IDCT to each row within the block
            for (int bi = 0; bi < block_rows; ++bi) {
                vector<double> row(block_cols);
                for (int bj = 0; bj < block_cols; ++bj)
                    row[bj] = rw::idct::image.at<float>(i + bi, j + bj);
                vector<double> idct_row = idct_1d(row);
                for (int bj = 0; bj < block_cols; ++bj)
                    rw::idct::image.at<float>(i + bi, j + bj) = idct_row[bj];
            }
        }
    }
    return 0;
}

// 2D-IDCT using two 1D-IDCTs
Mat idct_2d(const Mat& dct_matrix, const int& num_threads_assigned = 4, const string& partition_assigned = "block", const string& mode = "2D") {
    ro::rows        = dct_matrix.rows;
    ro::cols        = dct_matrix.cols;
    rw::idct::image = Mat(ro::rows, ro::cols, CV_32F, Scalar(0));
    ro::idct::dct_channel = dct_matrix;

    ro::num_threads = num_threads_assigned;
    ro::partition   = partition_assigned;

    ro::block_rows            = ro::rows / util::image::kBlockSize + (ro::rows % util::image::kBlockSize != 0);
    ro::block_rows_per_thread = ro::block_rows / ro::num_threads;
    ro::rows_per_thread       = ro::block_rows_per_thread * util::image::kBlockSize;

    vector<pthread_t> thread_handles(ro::num_threads - 1);

    for (long thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_create(&thread_handles[thread], nullptr, sub_idct_2d, (void*)thread);
    }
    for (int i = (ro::num_threads - 1) * ro::rows_per_thread; i < ro::rows; i += util::image::kBlockSize) {
        for (int j = 0; j < ro::cols; j += util::image::kBlockSize) {
            const int block_rows = min(util::image::kBlockSize, ro::rows - i);
            const int block_cols = min(util::image::kBlockSize, ro::cols - j);

            // Step 1: Apply 1D-IDCT to each column within the block
            for (int bj = 0; bj < block_cols; ++bj) {
                vector<double> col(block_rows);
                for (int bi = 0; bi < block_rows; ++bi)
                    col[bi] = dct_matrix.at<float>(i + bi, j + bj);
                vector<double> idct_col = idct_1d(col);
                for (int bi = 0; bi < block_rows; ++bi)
                    rw::idct::image.at<float>(i + bi, j + bj) = idct_col[bi];
            }

            // Step 2: Apply 1D-IDCT to each row within the block
            for (int bi = 0; bi < block_rows; ++bi) {
                vector<double> row(block_cols);
                for (int bj = 0; bj < block_cols; ++bj)
                    row[bj] = rw::idct::image.at<float>(i + bi, j + bj);
                vector<double> idct_row = idct_1d(row);
                for (int bj = 0; bj < block_cols; ++bj)
                    rw::idct::image.at<float>(i + bi, j + bj) = idct_row[bj];
            }
        }
    }
    for (int thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_join(thread_handles[thread], nullptr);
    }

    return rw::idct::image;
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

void* sub_dct_2d(void* thread) {
    long       thread_id{long(thread)};
    const Mat& image{ro::dct::channel_data};
    for (int i = thread_id * ro::rows_per_thread; i < (thread_id + 1) * ro::rows_per_thread; i += util::image::kBlockSize) {
        for (int j = 0; j < ro::cols; j += util::image::kBlockSize) {
            const int block_rows = min(util::image::kBlockSize, ro::rows - i);
            const int block_cols = min(util::image::kBlockSize, ro::cols - j);

            // Step 1: Apply 1D-DCT to each row within the block
            for (int bi = 0; bi < block_rows; ++bi) {
                vector<double> row(block_cols);
                for (int bj = 0; bj < block_cols; ++bj)
                    row[bj] = image.at<float>(i + bi, j + bj);
                vector<double> dct_row = dct_1d(row);
                for (int bj = 0; bj < block_cols; ++bj)
                    rw::dct::dct_matrix.at<float>(i + bi, j + bj) = dct_row[bj];
            }

            // Step 2: Apply 1D-DCT to each column within the block
            for (int bj = 0; bj < block_cols; ++bj) {
                vector<double> col(block_rows);
                for (int bi = 0; bi < block_rows; ++bi)
                    col[bi] = rw::dct::dct_matrix.at<float>(i + bi, j + bj);
                vector<double> dct_col = dct_1d(col);
                for (int bi = 0; bi < block_rows; ++bi)
                    rw::dct::dct_matrix.at<float>(i + bi, j + bj) = dct_col[bi];
            }
        }
    }
    return 0;
}

// 2D-DCT using two 1D-DCTs
Mat dct_2d(const Mat& image, const int& num_threads_assigned = 4, const string& partition_assigned = "block", const string& mode = "2D") {
    ro::rows            = image.rows;
    ro::cols            = image.cols;
    rw::dct::dct_matrix = Mat(ro::rows, ro::cols, CV_32F);
    ro::dct::channel_data = image;

    ro::num_threads = num_threads_assigned;
    ro::partition   = partition_assigned;

    ro::block_rows            = ro::rows / util::image::kBlockSize + (ro::rows % util::image::kBlockSize != 0);
    ro::block_rows_per_thread = ro::block_rows / ro::num_threads;
    ro::rows_per_thread       = ro::block_rows_per_thread * util::image::kBlockSize;

    vector<pthread_t> thread_handles(ro::num_threads - 1);

    for (long thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_create(&thread_handles[thread], nullptr, sub_dct_2d, (void*)thread);
    }
    for (int i = (ro::num_threads - 1) * ro::rows_per_thread; i < ro::rows; i += util::image::kBlockSize) {
        for (int j = 0; j < ro::cols; j += util::image::kBlockSize) {
            const int block_rows = min(util::image::kBlockSize, ro::rows - i);
            const int block_cols = min(util::image::kBlockSize, ro::cols - j);

            // Step 1: Apply 1D-DCT to each row within the block
            for (int bi = 0; bi < block_rows; ++bi) {
                vector<double> row(block_cols);
                for (int bj = 0; bj < block_cols; ++bj)
                    row[bj] = image.at<float>(i + bi, j + bj);
                vector<double> dct_row = dct_1d(row);
                for (int bj = 0; bj < block_cols; ++bj)
                    rw::dct::dct_matrix.at<float>(i + bi, j + bj) = dct_row[bj];
            }

            // Step 2: Apply 1D-DCT to each column within the block
            for (int bj = 0; bj < block_cols; ++bj) {
                vector<double> col(block_rows);
                for (int bi = 0; bi < block_rows; ++bi)
                    col[bi] = rw::dct::dct_matrix.at<float>(i + bi, j + bj);
                vector<double> dct_col = dct_1d(col);
                for (int bi = 0; bi < block_rows; ++bi)
                    rw::dct::dct_matrix.at<float>(i + bi, j + bj) = dct_col[bi];
            }
        }
    }
    for (int thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_join(thread_handles[thread], nullptr);
    }

    return rw::dct::dct_matrix;
}

void dct_3d(const util::image::Channel3d& original, util::image::Channel3d& dct, const int& num_threads_assigned = 4) {
    for (int i{0}; i < 3; ++i) {
        dct[i] = dct_2d(original[i], num_threads_assigned);
    }
}

void idct_3d(const util::image::Channel3d& dct, util::image::Channel3d& reconstructed, const int& num_threads_assigned = 4) {
    for (int i{0}; i < 3; ++i) {
        reconstructed[i] = idct_2d(dct[i], num_threads_assigned);
    }
}

void dct_4d(const vector<util::image::Channel3d>& originals, vector<util::image::Channel3d>& dcts, const int& num_threads_assigned = 4) {
    for (int i{0}; i < originals.size(); ++i) {
        dct_3d(originals[i], dcts[i], num_threads_assigned);
    }
}

void idct_4d(const vector<util::image::Channel3d>& dcts, vector<util::image::Channel3d>& reconstructeds, const int& num_threads_assigned = 4) {
    for (int i{0}; i < dcts.size(); ++i) {
        idct_3d(dcts[i], reconstructeds[i], num_threads_assigned);
    }
}

}  // namespace dct_pthread
