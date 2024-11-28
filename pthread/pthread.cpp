#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <chrono>

#include <pthread.h>

using namespace std;
using namespace cv;

// direction of 1D
namespace direction {
constexpr int kRow{0};
constexpr int kCol{1};
}  // namespace direction

// read-only
namespace ro {
namespace idct {
Mat dct_channel;
}  // namespace idct

namespace dct {
Mat channel_data;
}  // namespace dct

int    rows;
int    cols;
int    rows_per_thread;
int    cols_per_thread;
int    num_threads;
string partition;
int    outer_loop_direction;
}  // namespace ro

// read-write
namespace rw {
namespace idct {
Mat image;
}  // namespace idct

namespace dct {
Mat dct_matrix;
}  // namespace dct
}  // namespace rw

map<string, string> parse_args(int argc, char* argv[]) {
    map<string, string> args;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg.find("--")) continue;
        args[arg.substr(2)] = i + 1 < argc && string(argv[i + 1]).find("-") != 0 ? argv[++i] : "";
    }
    return args;
}

// 1D-IDCT
vector<double> idct_1d(const vector<double>& signal) {
    int N = signal.size();
    vector<double> result(N, 0.0);
    for (int x = 0; x < N; ++x) {
        double sum_value = 0.0;
        for (int u = 0; u < N; ++u) {
            double alpha_u = (u == 0) ? 1.0 / sqrt(N) : sqrt(2.0 / N);
            sum_value += alpha_u * signal[u] * cos(M_PI * (2 * x + 1) * u / (2 * N));
        }
        result[x] = sum_value;
    }
    return result;
}

void* sub_idct_2d(void* thread) {
    int        thread_id{long(thread)};
    const Mat& dct_matrix{ro::idct::dct_channel};
    if (ro::outer_loop_direction == direction::kRow) {
        for (int i = thread_id * ro::rows_per_thread; i < thread_id * ro::rows_per_thread + ro::rows_per_thread; ++i) {
            vector<double> row(ro::cols);
            for (int j = 0; j < ro::cols; ++j)
                row[j] = rw::idct::image.at<float>(i, j);
            vector<double> idct_row = idct_1d(row);
            for (int j = 0; j < ro::cols; ++j)
                rw::idct::image.at<float>(i, j) = idct_row[j];
        }
    } else {
        for (int j = thread_id * ro::cols_per_thread; j < thread_id * ro::cols_per_thread + ro::cols_per_thread; ++j) {
            vector<double> col(ro::rows);
            for (int i = 0; i < ro::rows; ++i)
                col[i] = dct_matrix.at<float>(i, j);
            vector<double> idct_col = idct_1d(col);
            for (int i = 0; i < ro::rows; ++i)
                rw::idct::image.at<float>(i, j) = idct_col[i];
        }
    }
}

// 2D-IDCT using two 1D-IDCTs
Mat idct_2d(const Mat& dct_matrix, const int& num_threads_assigned = 4, const string& partition_assigned = "block", const string& mode = "2D") {
    ro::rows = dct_matrix.rows;
    ro::cols = dct_matrix.cols;
    rw::idct::image = Mat(ro::rows, ro::cols, CV_32F, Scalar(0));

    ro::num_threads = num_threads_assigned;
    ro::partition   = partition_assigned;

    ro::rows_per_thread = ro::rows / ro::num_threads;
    ro::cols_per_thread = ro::cols / ro::num_threads;

    vector<pthread_t> thread_handles(ro::num_threads - 1);

    // Step 1: Apply 1D-IDCT to each column
    ro::outer_loop_direction = direction::kCol;
    for (long thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_create(&thread_handles[thread], nullptr, sub_idct_2d, (void*)thread);
    }
    for (int j = (ro::num_threads - 1) * ro::cols_per_thread; j < ro::cols; ++j) {
        vector<double> col(ro::rows);
        for (int i = 0; i < ro::rows; ++i)
            col[i] = dct_matrix.at<float>(i, j);
        vector<double> idct_col = idct_1d(col);
        for (int i = 0; i < ro::rows; ++i)
            rw::idct::image.at<float>(i, j) = idct_col[i];
    }
    for (int thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_join(thread_handles[thread], nullptr);
    }

    // Step 2: Apply 1D-IDCT to each row
    ro::outer_loop_direction = direction::kRow;
    for (long thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_create(&thread_handles[thread], nullptr, sub_idct_2d, (void*)thread);
    }
    for (int i = (ro::num_threads - 1) * ro::rows_per_thread; i < ro::rows; ++i) {
        vector<double> row(ro::cols);
        for (int j = 0; j < ro::cols; ++j)
            row[j] = rw::idct::image.at<float>(i, j);
        vector<double> idct_row = idct_1d(row);
        for (int j = 0; j < ro::cols; ++j)
            rw::idct::image.at<float>(i, j) = idct_row[j];
    }
    for (int thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_join(thread_handles[thread], nullptr);
    }

    return rw::idct::image;
}

// 1D-DCT
vector<double> dct_1d(const vector<double>& signal) {
    int N = signal.size();
    vector<double> result(N, 0.0);
    for (int u = 0; u < N; ++u) {
        double sum_value = 0.0;
        for (int x = 0; x < N; ++x) {
            sum_value += signal[x] * cos(M_PI * (2 * x + 1) * u / (2 * N));
        }
        result[u] = sum_value * ((u == 0) ? 1 / sqrt(N) : sqrt(2.0 / N));
    }
    return result;
}

void* sub_dct_2d(void* thread) {
    int        thread_id{long(thread)};
    const Mat& image{ro::dct::channel_data};
    if (ro::outer_loop_direction == direction::kRow) {
        for (int i = thread_id * ro::rows_per_thread; i < thread_id * ro::rows_per_thread + ro::rows_per_thread; ++i) {
            vector<double> row(ro::cols);
            for (int j = 0; j < ro::cols; ++j)
                row[j] = image.at<float>(i, j);
            vector<double> dct_row = dct_1d(row);
            for (int j = 0; j < ro::cols; ++j)
                rw::dct::dct_matrix.at<float>(i, j) = dct_row[j];
        }
    } else {
        for (int j = thread_id * ro::cols_per_thread; j < thread_id * ro::cols_per_thread + ro::cols_per_thread; ++j) {
            vector<double> col(ro::rows);
            for (int i = 0; i < ro::rows; ++i)
                col[i] = rw::dct::dct_matrix.at<float>(i, j);
            vector<double> dct_col = dct_1d(col);
            for (int i = 0; i < ro::rows; ++i)
                rw::dct::dct_matrix.at<float>(i, j) = dct_col[i];
        }
    }
}

// 2D-DCT using two 1D-DCTs
Mat dct_2d(const Mat& image, const int& num_threads_assigned = 4, const string& partition_assigned = "block", const string& mode = "2D") {
    ro::rows = image.rows;
    ro::cols = image.cols;
    rw::dct::dct_matrix = Mat(ro::rows, ro::cols, CV_32F);

    ro::num_threads = num_threads_assigned;
    ro::partition   = partition_assigned;

    ro::rows_per_thread = ro::rows / ro::num_threads;
    ro::cols_per_thread = ro::cols / ro::num_threads;

    vector<pthread_t> thread_handles(ro::num_threads - 1);

    // Step 1: Apply 1D-DCT to each row
    ro::outer_loop_direction = direction::kRow;
    for (long thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_create(&thread_handles[thread], nullptr, sub_dct_2d, (void*)thread);
    }
    for (int i = (ro::num_threads - 1) * ro::rows_per_thread; i < ro::rows; ++i) {
        vector<double> row(ro::cols);
        for (int j = 0; j < ro::cols; ++j)
            row[j] = image.at<float>(i, j);
        vector<double> dct_row = dct_1d(row);
        for (int j = 0; j < ro::cols; ++j)
            rw::dct::dct_matrix.at<float>(i, j) = dct_row[j];
    }
    for (int thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_join(thread_handles[thread], nullptr);
    }

    // Step 2: Apply 1D-DCT to each column
    ro::outer_loop_direction = direction::kCol;
    for (long thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_create(&thread_handles[thread], nullptr, sub_dct_2d, (void*)thread);
    }
    for (int j = (ro::num_threads - 1) * ro::cols_per_thread; j < ro::cols; ++j) {
        vector<double> col(ro::rows);
        for (int i = 0; i < ro::rows; ++i)
            col[i] = rw::dct::dct_matrix.at<float>(i, j);
        vector<double> dct_col = dct_1d(col);
        for (int i = 0; i < ro::rows; ++i)
            rw::dct::dct_matrix.at<float>(i, j) = dct_col[i];
    }
    for (int thread = 0; thread < ro::num_threads - 1; ++thread) {
        pthread_join(thread_handles[thread], nullptr);
    }

    return rw::dct::dct_matrix;
}

// PSNR Calculation
double calculate_psnr(const Mat& original, const Mat& reconstructed) {
    Mat diff;
    absdiff(original, reconstructed, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    double mse = sum(diff)[0] / (original.total());
    if (mse == 0) return 100;  // No error
    double max_pixel = 255.0;
    return 20.0 * log10(max_pixel / sqrt(mse));
}

int main(int argc, char* argv[]) {
    int num_threads = 4;
    string partition("block");
    string mode("2D");

    // load command-line arguments
    map<string, string> args = parse_args(argc, argv);
    if (args.find("num-threads") != args.end()) {
        num_threads = stoi(args["num-threads"]);
    }
    if (args.find("partition") != args.end()) {
        partition = args["partition"];
    }
    if (args.find("mode") != args.end()) {
        mode = args["mode"];
    }
    cout << "Number of threads: " << num_threads << endl;
    cout << "Partition: " << partition << endl;
    cout << "Mode: " << mode << endl;

    // Load the image
    Mat image = imread("lena.png", IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not load image." << endl;
        return -1;
    }
    resize(image, image, Size(256, 256));
    Mat image_float;
    image.convertTo(image_float, CV_32F);

    // Record start time for entire process
    auto total_start = chrono::high_resolution_clock::now();

    // Apply optimized 2D-DCT using two 1D-DCTs
    vector<Mat> dct_channels(3);
    auto dct_start = chrono::high_resolution_clock::now();
    for (int channel = 0; channel < 3; ++channel) {
        extractChannel(image_float, ro::dct::channel_data, channel);
        dct_channels[channel] = dct_2d(ro::dct::channel_data, num_threads, partition, mode);
    }
    auto dct_end = chrono::high_resolution_clock::now();
    cout << "Optimized 2D-DCT time: "
         << chrono::duration<double>(dct_end - dct_start).count() << " seconds" << endl;

    // Save the DCT image
    Mat dct_image;
    merge(dct_channels, dct_image);
    imwrite("output/dct_image.png", dct_image);

    // Apply optimized 2D-IDCT using two 1D-IDCTs
    auto idct_start = chrono::high_resolution_clock::now();
    vector<Mat> reconstructed_channels(3);
    for (int channel = 0; channel < 3; ++channel) {
        ro::idct::dct_channel = dct_channels[channel];
        reconstructed_channels[channel] = idct_2d(ro::idct::dct_channel, num_threads, partition, mode);
    }
    auto idct_end = chrono::high_resolution_clock::now();
    cout << "Optimized 2D-IDCT time: "
         << chrono::duration<double>(idct_end - idct_start).count() << " seconds" << endl;

    // Merge reconstructed channels and clip to valid range
    auto merge_start = chrono::high_resolution_clock::now();
    Mat reconstructed_image;
    merge(reconstructed_channels, reconstructed_image);
    reconstructed_image = min(max(reconstructed_image, 0.0f), 255.0f);
    reconstructed_image.convertTo(reconstructed_image, CV_8U);
    auto merge_end = chrono::high_resolution_clock::now();
    cout << "Merging and clipping time: "
         << chrono::duration<double>(merge_end - merge_start).count() << " seconds" << endl;

    // Save the reconstructed image
    imwrite("output/reconstructed_image.png", reconstructed_image);

    // Calculate PSNR
    auto psnr_start = chrono::high_resolution_clock::now();
    double psnr = calculate_psnr(image, reconstructed_image);
    auto psnr_end = chrono::high_resolution_clock::now();
    cout << "PSNR calculation time: "
         << chrono::duration<double>(psnr_end - psnr_start).count() << " seconds" << endl;
    cout << "PSNR: " << psnr << " dB" << endl;

    // Record end time for entire process
    auto total_end = chrono::high_resolution_clock::now();
    cout << "Total execution time: "
         << chrono::duration<double>(total_end - total_start).count() << " seconds" << endl;

    return 0;
}
