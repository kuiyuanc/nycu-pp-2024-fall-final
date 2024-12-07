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
    ExperimentArgs() = default;

    ExperimentArgs(map<string, string>& args) {
        datadir  = args.find("datadir") != args.end() ? args["datadir"] : "data/original";
        all_data = args.find("all-data") != args.end();

        image_size.first  = args.find("image-width") != args.end() ? std::stoi(args["image-width"]) : 256;
        image_size.second = args.find("image-height") != args.end() ? std::stoi(args["image-height"]) : 256;

        save = args.find("save") != args.end();

        num_threads = args.find("num-threads") != args.end() ? std::stoi(args["num-threads"]) : omp_get_max_threads();
        num_tests   = args.find("num-tests") != args.end() ? std::stoi(args["num-tests"]) : 30;

        tolerance = args.find("tolerance") != args.end() ? std::stod(args["tolerance"]) : 1e-5;

        verbose = args.find("verbose") != args.end();

        method = args.find("method") != args.end() ? args["method"] : "serial";
        if (method == "serial") {
            method_index = 0;
        } else if (method == "pthread") {
            method_index = 1;
        } else if (method == "omp") {
            method_index = 2;
        } else if (method == "cuda") {
            method_index = 3;
        }
    }

    string datadir{"data/original"};
    bool   all_data{false};

    util::image::Shape image_size{256, 256};

    int num_images{1};

    bool save{false};

    int num_threads{omp_get_max_threads()};
    int num_tests{30};

    double tolerance{1e-5};

    bool verbose{false};

    string method{"serial"};
    int    method_index{0};
};

class Experiment {
public:
    constexpr static short kLenSeparator{80};
    constexpr static short kNumMethods{4};
    constexpr static short kNumMeasurements{2};

    const array<string, kNumMethods>      kMethodNames{"Serial", "Pthread", "OpenMP", "CUDA"};
    const array<string, kNumMeasurements> kMeasurements{"DCT", "IDCT"};

    using BatchDCT  = function<void(const vector<util::image::Channel3d>&, vector<util::image::Channel3d>&, const int&)>;
    using BatchIDCT = BatchDCT;
    const array<BatchDCT, kNumMethods>  dcts{dct_serial::dct_4d, dct_pthread::dct_4d, dct_omp::dct_4d, dct_cuda::dct_4d};
    const array<BatchIDCT, kNumMethods> idcts{dct_serial::idct_4d, dct_pthread::idct_4d, dct_omp::idct_4d, dct_cuda::idct_4d};

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
    void load();
    void save();
    void test();
    void validate();
    void print() const;
    void print_args() const;
    void print_separator() const;

    bool last_all_data{false};
    ExperimentArgs args;

    vector<string>                                      filenames;
    vector<Mat>                                         original_images, dct_images, reconstructed_images;
    vector<util::image::Channel3d>                      original_channels, dct_channels, reconstructed_channels;
    array<vector<double>, Experiment::kNumMeasurements> time_elapsed;
    vector<double>                                      psnrs;
    bool                                                valid;
};

void Experiment::run() {
    double start;

    print_separator();

    if (args.verbose) {
        cout << "experiment starts" << endl;
        start = CycleTimer::currentSeconds();
    }

    // 1. load image(s)
    if (last_all_data != args.all_data || filenames.empty())
        load();

    if (args.verbose) {
        cout << "image(s) loaded in " << std::fixed << setprecision(3) << CycleTimer::currentSeconds() - start << " s" << endl;
        start = CycleTimer::currentSeconds();
    }

    // 2. test different dct & idct implementations
    test();

    if (args.verbose) {
        cout << "test(s) completed in " << std::fixed << setprecision(3) << CycleTimer::currentSeconds() - start << " s" << endl;
        start = CycleTimer::currentSeconds();
    }

    // 3. validate result(s)
    validate();

    if (args.verbose) {
        cout << "result(s) validated in " << std::fixed << setprecision(3) << CycleTimer::currentSeconds() - start << " s" << endl;
        print_separator();
    }

    // 4. print results
    print_args();
    print_separator();
    print();
    print_separator();

    // 5. record last arguments
    last_all_data = args.all_data;
}

void Experiment::load() {
    filenames = args.all_data ? util::system::get_filenames(args.datadir) : vector<string>{args.datadir + "/lena.png"};

    original_images = util::image::load(filenames, args.image_size);

    args.num_images = original_images.size();

    original_channels = util::image::split(original_images);

    for_each(execution::par_unseq, filenames.begin(), filenames.end(), [&](string& filename) { filename = filename.substr(args.datadir.length() + 1); });
}

void Experiment::save() {
    if (args.save) {
        dct_images.resize(args.num_images);
        util::image::merge(dct_channels, dct_images);
        util::image::save("data/dct/" + kMethodNames[args.method_index], filenames, dct_images);
        util::image::save("data/reconstructed/" + kMethodNames[args.method_index], filenames, reconstructed_images);
        cout << "Intermediate data are saved to data/dct and data/reconstructed" << endl;
    } else {
        cout << "Intermediate data are discarded" << endl;
    }
}

void Experiment::test() {
    for_each(time_elapsed.begin(), time_elapsed.end(), [&](vector<double>& times) { times.resize(args.num_tests); });
    dct_channels.resize(args.num_images);
    reconstructed_channels.resize(args.num_images);
    reconstructed_images.resize(args.num_images);

    for (size_t t{0}; t < args.num_tests; ++t) {
        time_elapsed[0][t] = util::system::timer(dcts[args.method_index], original_channels, dct_channels, args.num_threads);
        if (args.verbose) {
            cout << "test " << setw(log10(args.num_tests) + 1) << t + 1 << " DCT finished" << endl;
        }
        time_elapsed[1][t] = util::system::timer(idcts[args.method_index], dct_channels, reconstructed_channels, args.num_threads);
        if (args.verbose) {
            cout << "test " << setw(log10(args.num_tests) + 1) << t + 1 << " IDCT finished" << endl;
        }
        util::image::merge(reconstructed_channels, reconstructed_images);
        if (args.verbose) {
            cout << "test " << setw(log10(args.num_tests) + 1) << t + 1 << " finished" << endl;
        }
    }

    save();
}

void Experiment::validate() {
    valid = true;
    psnrs.resize(args.num_images);
    for (size_t i{0}; i < args.num_images; ++i) {
        psnrs[i] = util::image::calculate_psnr(original_images[i], reconstructed_images[i]);
        valid    = valid && 100.0 - psnrs[i] < args.tolerance;
    }
}

void Experiment::print() const {
    double lower, mean, upper;

    cout << std::showpoint << std::fixed;

    for (size_t i{0}; i < kNumMeasurements; ++i) {
        tie(lower, mean, upper) = util::statistics::ci95(time_elapsed[i]);
        cout << kMeasurements[i] << ':' << endl;
        cout << "\tMean:\t\t\t\t" << setw(8) << setprecision(5) << mean << " s" << endl;
        cout << "\t95% CI:\t\t\t[" << max(0.0, lower) << ", " << upper << "] s" << endl;
    }

    cout << "PSNR validation:\t" << (valid ? "Pass" : "Fail") << endl;
    cout << "PSNR mean:\t\t" << util::statistics::mean(psnrs) << endl;
}

void Experiment::print_args() const {
    cout << "Loading data from " << args.datadir << endl;
    if (args.all_data) {
        cout << '\t' << args.num_images << " images loaded." << endl;
    } else {
        cout << "\tLoading lena.png only" << endl;
    }
    cout << "\tImage shape: (" << args.image_size.first << ", " << args.image_size.second << ")" << endl;

    cout << "Testing with following parameters:" << endl;
    cout << "\tMethod: " << kMethodNames[args.method_index] << endl;
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
