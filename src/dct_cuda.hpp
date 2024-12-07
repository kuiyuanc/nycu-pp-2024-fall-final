#ifndef DCT_H
#define DCT_H

#include <vector>
#include <opencv2/opencv.hpp>

#include "lib/util.hpp"

using namespace cv;

#ifdef __CUDACC__  // Check if the file is being compiled by the CUDA compiler (nvcc)
__global__ void dctKernel(float* input, float* output, int width, int height);
__global__ void idctKernel(float* input, float* output, int width, int height);
#endif

namespace dct_cuda {

Mat  dct_2d(const Mat& image);
Mat  idct_2d(const Mat& dct_matrix);
void dct_3d(const util::image::Channel3d& original, util::image::Channel3d& dct);
void idct_3d(const util::image::Channel3d& dct, util::image::Channel3d& reconstructed);
void dct_4d(const vector<util::image::Channel3d>& originals, vector<util::image::Channel3d>& dcts, const int& num_threads_assigned = 4);
void idct_4d(const vector<util::image::Channel3d>& dcts, vector<util::image::Channel3d>& reconstructeds, const int& num_threads_assigned = 4);

}  // namespace dct_cuda

#endif  // DCT_H
