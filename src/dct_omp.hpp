#include <opencv2/opencv.hpp>
#include <vector>

#include "lib/util.hpp"

using namespace std;
using namespace cv;

#define BLOCK_SIZE 8

namespace dct_omp {

// 1D-IDCT
vector<double> dct_1d_omp(const vector<double> &signal) {
	const int N = signal.size();
	vector<double> result(N, 0.0);

	vector<vector<double>> cos_cache(N, vector<double>(N));
	for (int u = 0; u < N; ++u) {
		for (int x = 0; x < N; ++x) {
			cos_cache[u][x] = cos(M_PI * (2 * x + 1) * u / (2 * N));
		}
	}

	for (int u = 0; u < N; ++u) {
		double sum_value = 0.0;
		for (int x = 0; x < N; ++x) {
			sum_value += signal[x] * cos_cache[u][x];
		}
		result[u] = sum_value * ((u == 0) ? 1 / sqrt(N) : sqrt(2.0 / N));
	}
	return result;
}

Mat dct_2d(const Mat &image) {
	const int rows = image.rows;
	const int cols = image.cols;
	Mat dct_matrix(rows, cols, CV_32F);

	// Step 1: Block-wise row DCT
#pragma omp parallel for collapse(2) schedule(dynamic)
	for (int i = 0; i < rows; i += BLOCK_SIZE) {
		for (int j = 0; j < cols; j += BLOCK_SIZE) {
			const int block_rows = min(BLOCK_SIZE, rows - i);
			const int block_cols = min(BLOCK_SIZE, cols - j);
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

	Mat temp_matrix = dct_matrix.clone();

	// Step 2: Block-wise column DCT
#pragma omp parallel for collapse(2) schedule(dynamic)
	for (int j = 0; j < cols; j += BLOCK_SIZE) {
		for (int i = 0; i < rows; i += BLOCK_SIZE) {
			const int block_rows = min(BLOCK_SIZE, rows - i);
			const int block_cols = min(BLOCK_SIZE, cols - j);

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

void dct_3d(const util::image::Channel3d &image_channels, util::image::Channel3d &compressed_channels) {
	for (int channel = 0; channel < 3; ++channel) {
		compressed_channels[channel] = dct_2d(image_channels[channel]);
	}
}

void dct_4d(const vector<util::image::Channel3d> &originals, vector<util::image::Channel3d> &dcts,const int &num_threads_assigned = 4) {
	for (int i{0}; i < dcts.size(); ++i) {
		dct_3d(originals[i], dcts[i]);
	}
}

// 1D-IDCT
vector<double> idct_1d_omp(const vector<double> &signal) {
	const int N = signal.size();
	vector<double> result(N, 0.0);

	vector<vector<double>> cos_cache(N, vector<double>(N));
	vector<double> alpha_cache(N);

	for (int u = 0; u < N; ++u) {
		for (int x = 0; x < N; ++x) {
			cos_cache[u][x] = cos(M_PI * (2 * x + 1) * u / (2 * N));
		}
	}

	for (int u = 0; u < N; ++u) {
		alpha_cache[u] = (u == 0) ? 1.0 / sqrt(N) : sqrt(2.0 / N);
	}

	for (int x = 0; x < N; ++x) {
		double sum_value = 0.0;
		for (int u = 0; u < N; ++u) {
			sum_value += alpha_cache[u] * signal[u] * cos_cache[u][x];
		}
		result[x] = sum_value;
	}
	return result;
}

// 2D-IDCT
Mat idct_2d(const Mat &dct_matrix) {
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

void idct_3d(const util::image::Channel3d &compressed_channels, util::image::Channel3d &reconstructed_channels) {
	for (int channel = 0; channel < 3; ++channel) {
		reconstructed_channels[channel] = idct_2d(compressed_channels[channel]);
	}
}

void idct_4d(const vector<util::image::Channel3d> &dcts, vector<util::image::Channel3d> &reconstructed_channels,
			 const int &num_threads_assigned = 4) {
	for (int i{0}; i < dcts.size(); ++i) {
		idct_3d(dcts[i], reconstructed_channels[i]);
	}
}

} // namespace dct_omp
