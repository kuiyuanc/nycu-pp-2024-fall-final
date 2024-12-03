#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "dct_cuda.hpp"

using namespace cv;

__device__ float alpha(int u, int N) {
    return (u == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
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
                result += block[i][j] * cos((2 * i + 1) * threadIdx.y * M_PI / (2 * BLOCK_SIZE)) *
                          cos((2 * j + 1) * threadIdx.x * M_PI / (2 * BLOCK_SIZE));
            }
        }
        result *= alpha(threadIdx.y, BLOCK_SIZE) * alpha(threadIdx.x, BLOCK_SIZE);
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
                result += alpha(i, BLOCK_SIZE) * alpha(j, BLOCK_SIZE) *
                          block[i][j] * cos((2 * threadIdx.y + 1) * i * M_PI / (2 * BLOCK_SIZE)) *
                          cos((2 * threadIdx.x + 1) * j * M_PI / (2 * BLOCK_SIZE));
            }
        }
    }

    if (x < width && y < height) {
        output[y * width + x] = result;
    }
}

Mat dct_2d_cuda(const Mat& image) {
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

Mat idct_2d_cuda(const Mat& dct_matrix) {
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
