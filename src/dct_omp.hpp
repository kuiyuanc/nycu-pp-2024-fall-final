#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;

#define BLOCK_SIZE  8

// 優化 1D-DCT: 使用向量化和更好的快取利用
vector<double> dct_1d_omp(const vector<double>& signal) {
    const int N = signal.size();
    vector<double> result(N, 0.0);

    // 預計算 cosine 值以減少重複計算
    vector<vector<double>> cos_cache(N, vector<double>(N));
#pragma omp parallel for collapse(2)
    for (int u = 0; u < N; ++u) {
        for (int x = 0; x < N; ++x) {
            cos_cache[u][x] = cos(M_PI * (2 * x + 1) * u / (2 * N));
        }
    }
#pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < N; ++u) {
        double sum_value = 0.0;
#pragma omp simd reduction(+:sum_value)
        for (int x = 0; x < N; ++x) {
            sum_value += signal[x] * cos_cache[u][x];
        }
        result[u] = sum_value * ((u == 0) ? 1 / sqrt(N) : sqrt(2.0 / N));
    }
    return result;
}

// 使用 block 處理和更有效的資料存取
Mat dct_2d_omp(const Mat& image) {
    const int rows = image.rows;
    const int cols = image.cols;
    Mat dct_matrix(rows, cols, CV_32F);

    // 使用 block 處理來提高快取命中率


    // Step 1: Block-wise row DCT
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < rows; i += BLOCK_SIZE) {
        for (int j = 0; j < cols; j += BLOCK_SIZE) {
            const int block_rows = min(BLOCK_SIZE, rows - i);
            const int block_cols = min(BLOCK_SIZE, cols - j);

            // 處理當前 block 中的每一行
            for (int bi = 0; bi < block_rows; ++bi) {
                vector<double> row(block_cols);
                for (int bj = 0; bj < block_cols; ++bj) {
                    row[bj] = image.at<float>(i + bi, j + bj);
                }

                vector<double> dct_row = dct_1d_omp(row);
                for (int bj = 0; bj < block_cols; ++bj) {
                    dct_matrix.at<float>(i + bi, j + bj) = dct_row[bj];
                }
            }
        }
    }

    // 建立暫存矩陣以避免讀寫衝突
    Mat temp_matrix = dct_matrix.clone();

    // Step 2: Block-wise column DCT
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int j = 0; j < cols; j += BLOCK_SIZE) {
        for (int i = 0; i < rows; i += BLOCK_SIZE) {
            const int block_rows = min(BLOCK_SIZE, rows - i);
            const int block_cols = min(BLOCK_SIZE, cols - j);

            // 處理當前 block 中的每一列
            for (int bj = 0; bj < block_cols; ++bj) {
                vector<double> col(block_rows);
                for (int bi = 0; bi < block_rows; ++bi) {
                    col[bi] = temp_matrix.at<float>(i + bi, j + bj);
                }

                vector<double> dct_col = dct_1d_omp(col);
                for (int bi = 0; bi < block_rows; ++bi) {
                    dct_matrix.at<float>(i + bi, j + bj) = dct_col[bi];
                }
            }
        }
    }

    return dct_matrix;
}

// 優化 1D-IDCT
vector<double> idct_1d_omp(const vector<double>& signal) {
    const int N = signal.size();
    vector<double> result(N, 0.0);

    // 預計算 cosine 值和 alpha 值
    vector<vector<double>> cos_cache(N, vector<double>(N));
    vector<double> alpha_cache(N);

#pragma omp parallel for collapse(2)
    for (int u = 0; u < N; ++u) {
        for (int x = 0; x < N; ++x) {
            cos_cache[u][x] = cos(M_PI * (2 * x + 1) * u / (2 * N));
        }
    }

#pragma omp parallel for
    for (int u = 0; u < N; ++u) {
        alpha_cache[u] = (u == 0) ? 1.0 / sqrt(N) : sqrt(2.0 / N);
    }

#pragma omp parallel for schedule(dynamic)
    for (int x = 0; x < N; ++x) {
        double sum_value = 0.0;
#pragma omp simd reduction(+:sum_value)
        for (int u = 0; u < N; ++u) {
            sum_value += alpha_cache[u] * signal[u] * cos_cache[u][x];
        }
        result[x] = sum_value;
    }
    return result;
}

// 優化 2D-IDCT
Mat idct_2d_omp(const Mat& dct_matrix) {
    const int rows = dct_matrix.rows;
    const int cols = dct_matrix.cols;
    Mat image(rows, cols, CV_32F);


    // Step 1: Block-wise column IDCT
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int j = 0; j < cols; j += BLOCK_SIZE) {
        for (int i = 0; i < rows; i += BLOCK_SIZE) {
            const int block_rows = min(BLOCK_SIZE, rows - i);
            const int block_cols = min(BLOCK_SIZE, cols - j);

            for (int bj = 0; bj < block_cols; ++bj) {
                vector<double> col(block_rows);
                for (int bi = 0; bi < block_rows; ++bi) {
                    col[bi] = dct_matrix.at<float>(i + bi, j + bj);
                }

                vector<double> idct_col = idct_1d_omp(col);
                for (int bi = 0; bi < block_rows; ++bi) {
                    image.at<float>(i + bi, j + bj) = idct_col[bi];
                }
            }
        }
    }

    // 建立暫存矩陣
    Mat temp_image = image.clone();

    // Step 2: Block-wise row IDCT
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < rows; i += BLOCK_SIZE) {
        for (int j = 0; j < cols; j += BLOCK_SIZE) {
            const int block_rows = min(BLOCK_SIZE, rows - i);
            const int block_cols = min(BLOCK_SIZE, cols - j);

            for (int bi = 0; bi < block_rows; ++bi) {
                vector<double> row(block_cols);
                for (int bj = 0; bj < block_cols; ++bj) {
                    row[bj] = temp_image.at<float>(i + bi, j + bj);
                }

                vector<double> idct_row = idct_1d_omp(row);
                for (int bj = 0; bj < block_cols; ++bj) {
                    image.at<float>(i + bi, j + bj) = idct_row[bj];
                }
            }
        }
    }

    return image;
}


