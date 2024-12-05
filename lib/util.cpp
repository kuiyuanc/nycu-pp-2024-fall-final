#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "util.hpp"

using namespace std;
using namespace cv;

double util::statistics::mean(const vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double util::statistics::var(const vector<double>& values, bool population) {
    double avg             = mean(values);
    double sum_square_diff = accumulate(values.begin(), values.end(), 0.0, [&avg](const double& sum, const double& x) {
        return sum + (x - avg) * (x - avg);
    });
    return sum_square_diff / (population ? values.size() : values.size() - 1);
}

double util::statistics::stdev(const vector<double>& values, bool population) {
    return sqrt(var(values, population));
}

tuple<double, double, double> util::statistics::ci95(const vector<double>& values) {
    double avg   = mean(values);
    double sterr = stdev(values) / sqrt(values.size());
    return {avg - sterr * 1.96, avg, avg + sterr * 1.96};
}

// PSNR Calculation
double util::image::calculate_psnr(const Mat& original, const Mat& reconstructed) {
    // 確保圖像大小和通道一致
    if (original.size() != reconstructed.size() || original.channels() != reconstructed.channels()) {
        cerr << "Error: Input images must have the same size and number of channels." << endl;
        return -1;  // 返回錯誤值
    }

    Mat diff;
    absdiff(original, reconstructed, diff);  // 計算絕對差異
    diff.convertTo(diff, CV_32F);            // 確保數據為浮點型
    diff = diff.mul(diff);                   // 計算平方

    // 對多通道圖像進行處理
    Scalar channel_sum = sum(diff);  // 每個通道的平方和
    double mse         = 0.0;
    for (int i = 0; i < original.channels(); ++i) {
        mse += channel_sum[i];  // 累加所有通道的誤差
    }
    mse /= (original.total() * original.channels());  // 平均均方誤差

    if (mse == 0) return 100.0;  // 無誤差時返回最大 PSNR 值
    double max_pixel = 255.0;    // 最大像素值，適用於 8-bit 圖像
    return 20.0 * log10(max_pixel / sqrt(mse));
}

Mat util::image::load(string filename, const Shape& image_size) {
    Mat image = imread(filename, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not load image \'" << filename << '\'' << endl;
        exit(-1);
    }
    resize(image, image, Size(image_size.first, image_size.second));
    return image;
}

vector<Mat> util::image::load(const vector<string>& filenames, const Shape& image_size) {
    vector<Mat> images;
    for_each(filenames.begin(), filenames.end(), [&images, &image_size](const string& filename) {
        if (filename.rfind(".png") != string::npos) images.emplace_back(load(filename, image_size));
    });
    return images;
}

vector<util::image::Channel3d> util::image::split(const vector<Mat>& images) {
    vector<util::image::Channel3d> channels(images.size());
    Mat                            image_float;
    for (size_t i{0}; i < images.size(); ++i) {
        images[i].convertTo(image_float, CV_32F);
        cv::split(image_float, channels[i]);
    }
    return channels;
}

void util::image::merge(const vector<util::image::Channel3d>& channels, vector<Mat>& images) {
    images.resize(channels.size());
    for (size_t i{0}; i < images.size(); ++i) {
        cv::merge(channels[i], images[i]);
        images[i] = min(max(images[i], 0.0f), 255.0f);
        images[i].convertTo(images[i], CV_8U);
    }
}

void util::image::save(const string& datadir, const vector<string>& filenames, const vector<Mat>& images) {
    for (size_t i{0}; i < images.size(); ++i) {
        imwrite(datadir + "/" + filenames[i], images[i]);
    }
}

map<string, string> util::system::parse_args(int argc, char* argv[]) {
    vector<string>      argv_string(argv + 1, argv + argc);
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

vector<string> util::system::get_filenames(const string& datadir) {
    vector<string> filenames;
    for (const auto& entry : filesystem::directory_iterator(datadir)) {
        filenames.emplace_back(entry.path());
    }
    return filenames;
}
