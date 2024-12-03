#pragma once

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace util {
// PSNR Calculation
double calculate_psnr(const Mat& original, const Mat& reconstructed) {
    Mat diff;
    absdiff(original, reconstructed, diff);
    diff.convertTo(diff, CV_32F);
    diff       = diff.mul(diff);
    double mse = sum(diff)[0] / (original.total());
    if (mse == 0) return 100;  // No error
    double max_pixel = 255.0;
    return 20.0 * log10(max_pixel / sqrt(mse));
}

map<string, string> parse_args(int argc, char* argv[]) {
    map<string, string> args;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg.find("--")) continue;
        args[arg.substr(2)] = i + 1 < argc && string(argv[i + 1]).find("-") != 0 ? argv[++i] : "";
    }
    return args;
}
}  // namespace util
