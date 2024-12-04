#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

#include <pthread.h>

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
int    rows_per_thread;
int    cols_per_thread;
int    num_threads;
string partition;
int    outer_loop_direction;
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
            double alpha_u = (u == 0) ? 1.0 / sqrt(N) : sqrt(2.0 / N);
            sum_value += alpha_u * signal[u] * cos(M_PI * (2 * x + 1) * u / (2 * N));
        }
        result[x] = sum_value;
    }
    return result;
}

void* sub_idct_2d(void* thread) {
    long       thread_id{long(thread)};
    const Mat& dct_matrix{ro::idct::dct_channel};
    if (ro::outer_loop_direction == direction::kRow) {
        for (int i = thread_id * ro::rows_per_thread; i < thread_id * ro::rows_per_thread + ro::rows_per_thread; ++i) {
            vector<double> row(ro::cols);
            for (int j = 0; j < ro::cols; ++j)
                row[j] = rw::idct::image.at<float>(i, j);
            vector<double> idct_row = idct_1d(row);
            for (int j = 0; j < ro::cols; ++j)
                rw::idct::image.at<float>(i, j) = idct_row[j];
        }
    } else {
        for (int j = thread_id * ro::cols_per_thread; j < thread_id * ro::cols_per_thread + ro::cols_per_thread; ++j) {
            vector<double> col(ro::rows);
            for (int i = 0; i < ro::rows; ++i)
                col[i] = dct_matrix.at<float>(i, j);
            vector<double> idct_col = idct_1d(col);
            for (int i = 0; i < ro::rows; ++i)
                rw::idct::image.at<float>(i, j) = idct_col[i];
        }
    }
    return 0;
}

// 2D-IDCT using two 1D-IDCTs
Mat idct_2d(const Mat& dct_matrix, const int& num_threads_assigned = 4, const string& partition_assigned = "block", const string& mode = "2D") {
    ro::rows        = dct_matrix.rows;
    ro::cols        = dct_matrix.cols;
    rw::idct::image = Mat(ro::rows, ro::cols, CV_32F, Scalar(0));

    ro::num_threads = num_threads_assigned;
    ro::partition   = partition_assigned;

    ro::rows_per_thread = ro::rows / ro::num_threads;
    ro::cols_per_thread = ro::cols / ro::num_threads;

    vector<pthread_t> thread_handles(ro::num_threads - 1);

    // Step 1: Apply 1D-IDCT to each column
    ro::outer_loop_direction = direction::kCol;
    for (long thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_create(&thread_handles[thread], nullptr, sub_idct_2d, (void*)thread);
    }
    for (int j = (ro::num_threads - 1) * ro::cols_per_thread; j < ro::cols; ++j) {
        vector<double> col(ro::rows);
        for (int i = 0; i < ro::rows; ++i)
            col[i] = dct_matrix.at<float>(i, j);
        vector<double> idct_col = idct_1d(col);
        for (int i = 0; i < ro::rows; ++i)
            rw::idct::image.at<float>(i, j) = idct_col[i];
    }
    for (int thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_join(thread_handles[thread], nullptr);
    }

    // Step 2: Apply 1D-IDCT to each row
    ro::outer_loop_direction = direction::kRow;
    for (long thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_create(&thread_handles[thread], nullptr, sub_idct_2d, (void*)thread);
    }
    for (int i = (ro::num_threads - 1) * ro::rows_per_thread; i < ro::rows; ++i) {
        vector<double> row(ro::cols);
        for (int j = 0; j < ro::cols; ++j)
            row[j] = rw::idct::image.at<float>(i, j);
        vector<double> idct_row = idct_1d(row);
        for (int j = 0; j < ro::cols; ++j)
            rw::idct::image.at<float>(i, j) = idct_row[j];
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
            sum_value += signal[x] * cos(M_PI * (2 * x + 1) * u / (2 * N));
        }
        result[u] = sum_value * ((u == 0) ? 1 / sqrt(N) : sqrt(2.0 / N));
    }
    return result;
}

void* sub_dct_2d(void* thread) {
    long       thread_id{long(thread)};
    const Mat& image{ro::dct::channel_data};
    if (ro::outer_loop_direction == direction::kRow) {
        for (int i = thread_id * ro::rows_per_thread; i < thread_id * ro::rows_per_thread + ro::rows_per_thread; ++i) {
            vector<double> row(ro::cols);
            for (int j = 0; j < ro::cols; ++j)
                row[j] = image.at<float>(i, j);
            vector<double> dct_row = dct_1d(row);
            for (int j = 0; j < ro::cols; ++j)
                rw::dct::dct_matrix.at<float>(i, j) = dct_row[j];
        }
    } else {
        for (int j = thread_id * ro::cols_per_thread; j < thread_id * ro::cols_per_thread + ro::cols_per_thread; ++j) {
            vector<double> col(ro::rows);
            for (int i = 0; i < ro::rows; ++i)
                col[i] = rw::dct::dct_matrix.at<float>(i, j);
            vector<double> dct_col = dct_1d(col);
            for (int i = 0; i < ro::rows; ++i)
                rw::dct::dct_matrix.at<float>(i, j) = dct_col[i];
        }
    }
    return 0;
}

// 2D-DCT using two 1D-DCTs
Mat dct_2d(const Mat& image, const int& num_threads_assigned = 4, const string& partition_assigned = "block", const string& mode = "2D") {
    ro::rows            = image.rows;
    ro::cols            = image.cols;
    rw::dct::dct_matrix = Mat(ro::rows, ro::cols, CV_32F);

    ro::num_threads = num_threads_assigned;
    ro::partition   = partition_assigned;

    ro::rows_per_thread = ro::rows / ro::num_threads;
    ro::cols_per_thread = ro::cols / ro::num_threads;

    vector<pthread_t> thread_handles(ro::num_threads - 1);

    // Step 1: Apply 1D-DCT to each row
    ro::outer_loop_direction = direction::kRow;
    for (long thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_create(&thread_handles[thread], nullptr, sub_dct_2d, (void*)thread);
    }
    for (int i = (ro::num_threads - 1) * ro::rows_per_thread; i < ro::rows; ++i) {
        vector<double> row(ro::cols);
        for (int j = 0; j < ro::cols; ++j)
            row[j] = image.at<float>(i, j);
        vector<double> dct_row = dct_1d(row);
        for (int j = 0; j < ro::cols; ++j)
            rw::dct::dct_matrix.at<float>(i, j) = dct_row[j];
    }
    for (int thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_join(thread_handles[thread], nullptr);
    }

    // Step 2: Apply 1D-DCT to each column
    ro::outer_loop_direction = direction::kCol;
    for (long thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_create(&thread_handles[thread], nullptr, sub_dct_2d, (void*)thread);
    }
    for (int j = (ro::num_threads - 1) * ro::cols_per_thread; j < ro::cols; ++j) {
        vector<double> col(ro::rows);
        for (int i = 0; i < ro::rows; ++i)
            col[i] = rw::dct::dct_matrix.at<float>(i, j);
        vector<double> dct_col = dct_1d(col);
        for (int i = 0; i < ro::rows; ++i)
            rw::dct::dct_matrix.at<float>(i, j) = dct_col[i];
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
