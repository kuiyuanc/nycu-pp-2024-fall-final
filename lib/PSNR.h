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