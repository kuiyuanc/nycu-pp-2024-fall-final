#include <iostream>
#include <vector>
#include <cmath>
#include <FreeImage.h>
#include <chrono>

using namespace std;

// Helper function to calculate 1D-DCT
vector<double> dct_1d(const vector<double>& signal) {
    int N = signal.size();
    vector<double> result(N, 0.0);

    for (int u = 0; u < N; ++u) {
        double sum_value = 0.0;
        for (int x = 0; x < N; ++x) {
            sum_value += signal[x] * cos((M_PI * (2 * x + 1) * u) / (2 * N));
        }
        result[u] = (u == 0 ? (1 / sqrt(N)) : sqrt(2.0 / N)) * sum_value;
    }
    return result;
}

// Helper function to calculate 1D-IDCT
vector<double> idct_1d(const vector<double>& signal) {
    int N = signal.size();
    vector<double> result(N, 0.0);

    for (int x = 0; x < N; ++x) {
        double sum_value = 0.0;
        for (int u = 0; u < N; ++u) {
            double alpha_u = (u == 0 ? (1 / sqrt(N)) : sqrt(2.0 / N));
            sum_value += alpha_u * signal[u] * cos((M_PI * (2 * x + 1) * u) / (2 * N));
        }
        result[x] = sum_value;
    }
    return result;
}

// Helper function to calculate 2D-DCT using two 1D-DCTs
vector<vector<double>> dct_2d(const vector<vector<double>>& image) {
    int N = image.size(), M = image[0].size();
    vector<vector<double>> dct_matrix(N, vector<double>(M, 0.0));

    // Apply 1D-DCT to rows
    for (int i = 0; i < N; ++i) {
        dct_matrix[i] = dct_1d(image[i]);
    }

    // Apply 1D-DCT to columns
    for (int j = 0; j < M; ++j) {
        vector<double> column(N);
        for (int i = 0; i < N; ++i) column[i] = dct_matrix[i][j];
        column = dct_1d(column);
        for (int i = 0; i < N; ++i) dct_matrix[i][j] = column[i];
    }

    return dct_matrix;
}

// Helper function to calculate 2D-IDCT using two 1D-IDCTs
vector<vector<double>> idct_2d(const vector<vector<double>>& dct_matrix) {
    int N = dct_matrix.size(), M = dct_matrix[0].size();
    vector<vector<double>> image(N, vector<double>(M, 0.0));

    // Apply 1D-IDCT to columns
    for (int j = 0; j < M; ++j) {
        vector<double> column(N);
        for (int i = 0; i < N; ++i) column[i] = dct_matrix[i][j];
        column = idct_1d(column);
        for (int i = 0; i < N; ++i) image[i][j] = column[i];
    }

    // Apply 1D-IDCT to rows
    for (int i = 0; i < N; ++i) {
        image[i] = idct_1d(image[i]);
    }

    return image;
}

// Calculate PSNR between two images
double calculate_psnr(const vector<vector<double>>& original, const vector<vector<double>>& reconstructed) {
    double mse = 0.0;
    int N = original.size(), M = original[0].size();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            mse += pow(original[i][j] - reconstructed[i][j], 2);
        }
    }
    mse /= (N * M);

    if (mse == 0) return 100.0;
    double max_pixel = 255.0;
    return 20.0 * log10(max_pixel / sqrt(mse));
}

// Load image using FreeImage library
vector<vector<double>> load_image(const char* filename) {
    FreeImage_Initialise();
    FIBITMAP* bitmap = FreeImage_Load(FIF_PNG, filename, 0);
    FIBITMAP* grayBitmap = FreeImage_ConvertToGreyscale(bitmap);
    int width = FreeImage_GetWidth(grayBitmap);
    int height = FreeImage_GetHeight(grayBitmap);

    vector<vector<double>> image(height, vector<double>(width, 0.0));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image[y][x] = FreeImage_GetPixelIndex(grayBitmap, x, y);
        }
    }

    FreeImage_Unload(grayBitmap);
    FreeImage_Unload(bitmap);
    FreeImage_DeInitialise();
    return image;
}

int main() {
    // Load and preprocess the image
    const char* filename = "lena.png";
    vector<vector<double>> image = load_image(filename);

    // Perform 2D-DCT
    auto start = chrono::high_resolution_clock::now();
    vector<vector<double>> dct_matrix = dct_2d(image);
    auto end = chrono::high_resolution_clock::now();
    cout << "2D-DCT computation time: " << chrono::duration<double>(end - start).count() << " seconds" << endl;

    // Perform 2D-IDCT
    start = chrono::high_resolution_clock::now();
    vector<vector<double>> reconstructed_image = idct_2d(dct_matrix);
    end = chrono::high_resolution_clock::now();
    cout << "2D-IDCT computation time: " << chrono::duration<double>(end - start).count() << " seconds" << endl;

    // Calculate PSNR
    double psnr = calculate_psnr(image, reconstructed_image);
    cout << "PSNR: " << psnr << " dB" << endl;

    return 0;
}
