#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace util {
namespace statistics {

auto mean(const auto& values) {
    return accumulate(values.begin(), values.end(), 0.0) / values.size();
}

auto var(const auto& values, bool population = false) {
    auto avg             = mean(values);
    auto sum_square_diff = accumulate(values.begin(), values.end(), 0.0, [&avg](const auto& sum, const auto& x) {
        return sum + (x - avg) * (x - avg);
    });
    return sum_square_diff / (population ? values.size() : values.size() - 1);
}

auto stdev(const auto& values, bool population = false) {
    return sqrt(var(values, population));
}

tuple<double, double, double> ci95(const auto& values) {
    auto avg   = mean(values);
    auto sterr = stdev(values) / sqrt(values.size());
    return {avg - sterr * 1.96, avg, avg + sterr * 1.96};
}

}  // namespace statistics

// PSNR Calculation
double calculate_psnr(const Mat& original, const Mat& reconstructed) {
    // 確保圖像大小和通道一致
    if (original.size() != reconstructed.size() || original.channels() != reconstructed.channels()) {
        cerr << "Error: Input images must have the same size and number of channels." << endl;
        return -1; // 返回錯誤值
    }

    Mat diff;
    absdiff(original, reconstructed, diff); // 計算絕對差異
    diff.convertTo(diff, CV_32F);           // 確保數據為浮點型
    diff = diff.mul(diff);                  // 計算平方

    // 對多通道圖像進行處理
    Scalar channel_sum = sum(diff); // 每個通道的平方和
    double mse = 0.0;
    for (int i = 0; i < original.channels(); ++i) {
        mse += channel_sum[i]; // 累加所有通道的誤差
    }
    mse /= (original.total() * original.channels()); // 平均均方誤差

    if (mse == 0) return 100.0; // 無誤差時返回最大 PSNR 值
    double max_pixel = 255.0;   // 最大像素值，適用於 8-bit 圖像
    return 20.0 * log10(max_pixel / sqrt(mse));
}

void load_image(string filename, vector<Mat>& image_channels) {
    Mat image = imread(filename, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not load image." << endl;
        exit(-1);
    }
    Mat image_float;
    image.convertTo(image_float, CV_32F);
    vector<Mat> channels(3);
    split(image_float, channels);
    image_channels = channels;
}

map<string, string> parse_args(int argc, char* argv[]) {
    vector<string> argv_string(argv + 1, argv + argc);
    map<string, string> args;
    for (auto i = begin(argv_string); i != end(argv_string); ++i) {
        if (i->substr(0, 2) != "--") continue;
        args[i->substr(2)] = "";

        if (i + 1 == end(argv_string) || (i + 1)->front() == '-') continue;
        args[i->substr(2)] = *(i + 1);
        ++i;
    }
    return args;
}

}  // namespace util
