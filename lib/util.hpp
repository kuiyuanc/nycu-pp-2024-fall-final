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

// PSNR Calculation
double            calculate_psnr(const Mat& original, const Mat& reconstructed);
Mat               load(string filename, const Shape& image_size);
vector<Mat>       load(const vector<string>& filenames, const Shape& image_size);
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
