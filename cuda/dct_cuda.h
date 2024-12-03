#ifndef DCT_H
#define DCT_H

#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;

#define BLOCK_SIZE 8
// fatest: 4

#ifdef __CUDACC__  // Check if the file is being compiled by the CUDA compiler (nvcc)
__global__ void dctKernel(float* input, float* output, int width, int height);
__global__ void idctKernel(float* input, float* output, int width, int height);
#endif

Mat dct_2d_cuda(const Mat& image);
Mat idct_2d_cuda(const Mat& dct_matrix);

#endif // DCT_H
