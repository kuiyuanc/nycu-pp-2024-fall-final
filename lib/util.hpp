#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "CycleTimer.h"

using namespace std;
using namespace cv;

namespace util {
namespace statistics {

double                        mean(const vector<double>& values);
double                        var(const vector<double>& values, bool population = false);
double                        stdev(const vector<double>& values, bool population = false);
tuple<double, double, double> ci95(const vector<double>& values);

}  // namespace statistics

namespace image {

constexpr short NUM_CHANNELS{3};

using Shape     = pair<int, int>;
using Channel3d = array<Mat, NUM_CHANNELS>;

constexpr int                                             kBlockSize{8};
static const array<array<double, kBlockSize>, kBlockSize> cos_cache = []() {
    array<array<double, kBlockSize>, kBlockSize> cos_cache;
    for (int u = 0; u < kBlockSize; ++u) {
        for (int x = 0; x < kBlockSize; ++x) {
            cos_cache[u][x] = cos(M_PI * (2 * x + 1) * u / (2 * kBlockSize));
        }
    }
    return cos_cache;
}();
static const array<double, kBlockSize> alpha_cache = []() {
    array<double, kBlockSize> alpha_cache;
    for (int u = 0; u < kBlockSize; ++u) {
        alpha_cache[u] = u ? sqrt(2.0 / kBlockSize) : 1.0 / sqrt(kBlockSize);
    }
    return alpha_cache;
}();

// PSNR Calculation
double            calculate_psnr(const Mat& original, const Mat& reconstructed);
Mat               load(const string& filename);
vector<Mat>       load(const vector<string>& filenames);
vector<Channel3d> split(const vector<Mat>& images);
void              merge(const vector<Channel3d>& channels, vector<Mat>& images);
void              save(const string& datadir, const vector<string>& filenames, const vector<Mat>& images);

}  // namespace image

namespace system {

map<string, string> parse_args(int argc, char* argv[]);
vector<string>      get_filenames(const string& datadir);

template<typename Func, typename... Args>
double timer(Func function, Args&... args) {
    double start = CycleTimer::currentSeconds();
    function(args...);
    return CycleTimer::currentSeconds() - start;
}

}  // namespace system

}  // namespace util

#define BLOCK_SIZE util::image::kBlockSize
