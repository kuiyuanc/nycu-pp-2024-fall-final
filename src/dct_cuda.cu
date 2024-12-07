#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "dct_cuda.hpp"

#include "lib/util.hpp"

using namespace cv;

// Change device constants to float
__constant__ float d_cos_cache[BLOCK_SIZE][BLOCK_SIZE];
__constant__ float d_alpha_cache[BLOCK_SIZE];

namespace dct_cuda {

void copy_cache_to_device() {
    // Convert cos_cache to float
    float h_cos_cache[BLOCK_SIZE][BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            h_cos_cache[i][j] = static_cast<float>(util::image::cos_cache[i][j]);
        }
    }

    // Convert alpha_cache to float
    float h_alpha_cache[BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        h_alpha_cache[i] = static_cast<float>(util::image::alpha_cache[i]);
    }

    // Copy to device constant memory
    cudaMemcpyToSymbol(d_cos_cache, h_cos_cache, sizeof(d_cos_cache));
    cudaMemcpyToSymbol(d_alpha_cache, h_alpha_cache, sizeof(d_alpha_cache));
}

__global__ void dctKernel(float* input, float* output, int width, int height) {
    __shared__ float block[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < width && y < height) {
        block[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    float result = 0.0f;
    if (threadIdx.x < BLOCK_SIZE && threadIdx.y < BLOCK_SIZE) {
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                result += block[i][j] * d_cos_cache[threadIdx.y][i] * d_cos_cache[threadIdx.x][j];
            }
        }
        result *= d_alpha_cache[threadIdx.y] * d_alpha_cache[threadIdx.x];
    }

    if (x < width && y < height) {
        output[y * width + x] = result;
    }
}

__global__ void idctKernel(float* input, float* output, int width, int height) {
    __shared__ float block[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < width && y < height) {
        block[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    float result = 0.0f;
    if (threadIdx.x < BLOCK_SIZE && threadIdx.y < BLOCK_SIZE) {
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                result += d_alpha_cache[i] * d_alpha_cache[j] *
                          block[i][j] * d_cos_cache[i][threadIdx.y] * d_cos_cache[j][threadIdx.x];
            }
        }
    }

    if (x < width && y < height) {
        output[y * width + x] = result;
    }
}

Mat dct_2d(const Mat& image) {
    const int rows = image.rows;
    const int cols = image.cols;

    float* d_input;
    float* d_output;
    size_t size = rows * cols * sizeof(float);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, image.ptr<float>(), size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    dctKernel<<<gridDim, blockDim>>>(d_input, d_output, cols, rows);

    Mat dct_matrix(rows, cols, CV_32F);
    cudaMemcpy(dct_matrix.ptr<float>(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return dct_matrix;
}

Mat idct_2d(const Mat& dct_matrix) {
    const int rows = dct_matrix.rows;
    const int cols = dct_matrix.cols;

    float* d_input;
    float* d_output;
    size_t size = rows * cols * sizeof(float);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, dct_matrix.ptr<float>(), size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    idctKernel<<<gridDim, blockDim>>>(d_input, d_output, cols, rows);

    Mat reconstructed_image(rows, cols, CV_32F);
    cudaMemcpy(reconstructed_image.ptr<float>(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return reconstructed_image;
}

void dct_3d(const util::image::Channel3d& original, util::image::Channel3d& dct) {
    for (int i = 0; i < 3; ++i) {
        dct[i] = dct_2d(original[i]);
    }
}

void idct_3d(const util::image::Channel3d& dct, util::image::Channel3d& reconstructed) {
    for (int i = 0; i < 3; ++i) {
        reconstructed[i] = idct_2d(dct[i]);
    }
}

void dct_4d(const std::vector<util::image::Channel3d>& originals, std::vector<util::image::Channel3d>& dcts, const int& num_threads_assigned) {
    for (size_t i = 0; i < originals.size(); ++i) {
        dct_3d(originals[i], dcts[i]);
    }
}

void idct_4d(const std::vector<util::image::Channel3d>& dcts, std::vector<util::image::Channel3d>& reconstructeds, const int& num_threads_assigned) {
    for (size_t i = 0; i < dcts.size(); ++i) {
        idct_3d(dcts[i], reconstructeds[i]);
    }
}

}  // namespace dct_cuda