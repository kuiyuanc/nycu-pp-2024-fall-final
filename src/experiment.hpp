#include <array>
#include <execution>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <omp.h>
#include <opencv2/opencv.hpp>

#include "lib/util.hpp"

#include "dct_cuda.hpp"
#include "dct_omp.hpp"
#include "dct_pthread.hpp"
#include "dct_serial.hpp"

struct ExperimentArgs {
    // TODO: make parameters configurable
    // block size
    // partition mode
    // simd
    // cosine cache

    ExperimentArgs() = default;

    ExperimentArgs(map<string, string>& args) {
        datadir  = args.find("datadir") != args.end() ? args["datadir"] : "data/original";
        all_data = args.find("all-data") != args.end();

        image_size = args.find("image-size") != args.end() ? std::stoi(args["image-size"]) : 256;

        save = args.find("save") != args.end();

        num_threads = args.find("num-threads") != args.end() ? std::stoi(args["num-threads"]) : omp_get_max_threads();
        num_tests   = args.find("num-tests") != args.end() ? std::stoi(args["num-tests"]) : 30;

        tolerance = args.find("tolerance") != args.end() ? std::stod(args["tolerance"]) : 1e-5;
    }

    string datadir{"data/original"};
    bool   all_data{false};

    int image_size{256};

    int num_images{1};

    bool save{false};

    int num_threads{omp_get_max_threads()};
    int num_tests{30};

    double tolerance{1e-5};
};

class Experiment {
public:
    constexpr static short kLenSeparator{80};
    constexpr static short kNumMethods{4};
    constexpr static short kNumTests{2};

    const array<string, kNumMethods> kMethodNames{"Serial", "Pthread", "OpenMP", "CUDA"};
    const array<string, kNumTests>   tests{"DCT", "IDCT"};

    Experiment() = default;
    Experiment(map<string, string>& args) :
        args(args) {};
    Experiment(const ExperimentArgs& args) :
        args(args) {};
    void run();
    void run(const ExperimentArgs& args) { set_args(args); run(); }
    void set_args(const ExperimentArgs& args);
    void set_args(map<string, string>& args);

private:
    tuple<vector<Mat>, vector<util::image::Channel3d>, vector<string>> load();
    void save(const vector<string>& filenames, const vector<vector<util::image::Channel3d>>& dct_channels, const vector<vector<Mat>>& reconstructed_images);
    pair<vector<vector<Mat>>, vector<array<vector<double>, kNumTests>>> test(const vector<util::image::Channel3d>& original_channels, const vector<string>& filenames);
    pair<vector<bool>, vector<vector<double>>> validate(const vector<Mat>& original_images, const vector<vector<Mat>>& reconstructed_images);
    void print(const vector<array<vector<double>, kNumTests>>& time_elapsed, const vector<bool>& validations, const vector<vector<double>>& psnrs) const;
    void print_args() const;
    void print_separator() const;

    ExperimentArgs args;
};

void Experiment::run() {
    print_separator();

    // 1. load image(s)
    auto [original_images, original_channels, filenames] = load();

    // 2. test different dct & idct implementations
    auto [reconstructed_images, time_elapsed] = test(original_channels, filenames);

    // 3. validate result(s)
    auto [validations, psnrs] = validate(original_images, reconstructed_images);

    // 4. print results
    print_args();
    print_separator();
    print(time_elapsed, validations, psnrs);
    print_separator();
}

tuple<vector<Mat>, vector<util::image::Channel3d>, vector<string>> Experiment::load() {
    vector<string> filenames(args.all_data ? util::system::get_filenames(args.datadir) : vector<string>{args.datadir + "/lena.png"});

    vector<Mat> images(util::image::load(filenames, util::image::Shape{args.image_size, args.image_size}));

    args.num_images = images.size();

    vector<util::image::Channel3d> channels(util::image::split(images));

    for_each(execution::par_unseq, filenames.begin(), filenames.end(), [&](string& filename) { filename = filename.substr(args.datadir.length() + 1); });

    return {images, channels, filenames};
}

void Experiment::save(const vector<string>& filenames, const vector<vector<util::image::Channel3d>>& dct_channels, const vector<vector<Mat>>& reconstructed_images) {
    if (args.save) {
        vector<Mat> dct_images(args.num_images);
        for (int i{0}; i < kNumMethods; ++i) {
            util::image::merge(dct_channels[i], dct_images);
            util::image::save("data/dct/" + kMethodNames[i], filenames, dct_images);
            util::image::save("data/reconstructed/" + kMethodNames[i], filenames, reconstructed_images[i]);
        }
        cout << "Intermediate data are saved to data/dct and data/reconstructed" << endl;
    } else {
        cout << "Intermediate data are discarded" << endl;
    }
}

pair<vector<vector<Mat>>, vector<array<vector<double>, Experiment::kNumTests>>>
Experiment::test(const vector<util::image::Channel3d>& original_channels, const vector<string>& filenames) {
    using BatchDCT                            = function<void(const vector<util::image::Channel3d>&, vector<util::image::Channel3d>&, const int&)>;
    using BatchIDCT                           = BatchDCT;
    const array<BatchDCT, kNumMethods>  dcts  = {dct_serial::dct_4d, dct_pthread::dct_4d, dct_omp::dct_4d, dct_cuda::dct_4d};
    const array<BatchIDCT, kNumMethods> idcts = {dct_serial::idct_4d, dct_pthread::idct_4d, dct_omp::idct_4d, dct_cuda::idct_4d};

    vector<vector<util::image::Channel3d>>   dct_channels(kNumMethods, vector<util::image::Channel3d>(args.num_images));
    vector<vector<util::image::Channel3d>>   reconstructed_channels(kNumMethods, vector<util::image::Channel3d>(args.num_images));
    vector<array<vector<double>, kNumTests>> time_elapsed(kNumMethods, {vector<double>(args.num_tests), vector<double>(args.num_tests)});
    vector<vector<Mat>>                      reconstructed_images(kNumMethods, vector<Mat>(args.num_images));

    for (size_t t{0}; t < args.num_tests; ++t) {
        for (size_t i{0}; i < kNumMethods; ++i) {
            time_elapsed[i][0][t] = util::system::timer(dcts[i], original_channels, dct_channels[i], args.num_threads);
            time_elapsed[i][1][t] = util::system::timer(idcts[i], dct_channels[i], reconstructed_channels[i], args.num_threads);
            util::image::merge(reconstructed_channels[i], reconstructed_images[i]);
        }
    }

    save(filenames, dct_channels, reconstructed_images);

    return {reconstructed_images, time_elapsed};
}

pair<vector<bool>, vector<vector<double>>> Experiment::validate(const vector<Mat>& original_images, const vector<vector<Mat>>& reconstructed_images) {
    vector<bool>           validations(kNumMethods, true);
    vector<vector<double>> psnrs(kNumMethods, vector<double>(args.num_images));
    for (size_t i{0}; i < validations.size(); ++i) {
        for (size_t j{0}; j < args.num_images; ++j) {
            psnrs[i][j]    = util::image::calculate_psnr(original_images[j], reconstructed_images[i][j]);
            validations[i] = validations[i] && 100.0 - psnrs[i][j] < args.tolerance;
        }
    }
    return {validations, psnrs};
}

void Experiment::print(const vector<array<vector<double>, Experiment::kNumTests>>& time_elapsed, const vector<bool>& validations, const vector<vector<double>>& psnrs) const {
    double lower, mean, upper;

    cout << std::showpoint << std::fixed;

    for (size_t i{0}; i < kNumMethods; ++i) {
        cout << kMethodNames[i] << ':' << endl;

        for (size_t j{0}; j < kNumTests; ++j) {
            tie(lower, mean, upper) = util::statistics::ci95(time_elapsed[i][j]);
            cout << '\t' << tests[j] << ':' << endl;
            cout << "\t\tMean:\t\t\t\t" << setprecision(3) << mean << "  s" << endl;
            cout << "\t\t95% CI:\t\t\t[" << max(0.0, lower) << ", " << upper << "] s" << endl;
            cout << "\t\tSpeedup ratio:\t\t\t" << setw(5) << setprecision(1) << util::statistics::mean(time_elapsed[0][j]) / util::statistics::mean(time_elapsed[i][j]) * 100 << "  %" << endl;
        }

        cout << endl;
        cout << "\tPSNR validation:\t" << (validations[i] ? "Pass" : "Fail") << endl;
        cout << "\tPSNR mean:\t\t" << util::statistics::mean(psnrs[i]) << endl;

        if (i < kNumMethods - 1)
            cout << endl;
    }
}

void Experiment::print_args() const {
    cout << "Loading data from " << args.datadir << endl;
    if (args.all_data) {
        cout << '\t' << args.num_images << " images loaded." << endl;
    } else {
        cout << "\tLoading lena.png only" << endl;
    }
    cout << "\tImage shape: (" << args.image_size << ", " << args.image_size << ")" << endl;

    cout << "Testing with following parameters:" << endl;
    cout << "\tUsing " << args.num_threads << " threads" << endl;
    cout << "\tNumber of tests = " << args.num_tests << endl;

    cout << std::setprecision(1) << std::scientific << "PSNR validation acceptance tolerance = " << args.tolerance << endl;
}

void Experiment::set_args(const ExperimentArgs& args) {
    this->args = args;
}

void Experiment::set_args(map<string, string>& args) {
    this->args = ExperimentArgs(args);
}

void Experiment::print_separator() const {
    cout << string(kLenSeparator, '=') << endl;
}
