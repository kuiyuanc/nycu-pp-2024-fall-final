import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm

# 1D-IDCT
def idct_1d(signal):
    N = len(signal)
    result = np.zeros(N)
    for x in range(N):
        sum_value = 0
        for u in range(N):
            alpha_u = 1 / math.sqrt(N) if u == 0 else math.sqrt(2 / N)
            sum_value += alpha_u * signal[u] * math.cos((math.pi * (2 * x + 1) * u) / (2 * N))
        result[x] = sum_value
    return result

# 2D-IDCT using two 1D-IDCTs
def idct_2d(dct_matrix):
    N, M = dct_matrix.shape
    image = np.zeros((N, M))

    # Step 1: Apply 1D-IDCT to each column
    for j in tqdm(range(M), desc="1D-IDCT on columns"):
        image[:, j] = idct_1d(dct_matrix[:, j])

    # Step 2: Apply 1D-IDCT to each row
    for i in tqdm(range(N), desc="1D-IDCT on rows"):
        image[i, :] = idct_1d(image[i, :])

    return image

# 1D-DCT (for rows and columns)
def dct_1d(signal):
    N = len(signal)
    result = np.zeros(N)
    for u in range(N):
        sum_value = 0
        for x in range(N):
            sum_value += signal[x] * math.cos((math.pi * (2 * x + 1) * u) / (2 * N))
        result[u] = sum_value * (1 / math.sqrt(N)) if u == 0 else sum_value * math.sqrt(2 / N)
    return result

# 2D-DCT using two 1D-DCTs
def dct_2d(image):
    N, M = image.shape
    dct_matrix = np.zeros((N, M))

    # Step 1: Apply 1D-DCT to each row
    for i in tqdm(range(N), desc="1D-DCT on rows"):
        dct_matrix[i, :] = dct_1d(image[i, :])

    # Step 2: Apply 1D-DCT to each column
    for j in tqdm(range(M), desc="1D-DCT on columns"):
        dct_matrix[:, j] = dct_1d(dct_matrix[:, j])

    return dct_matrix

# Calculate PSNR
def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100  # If no error, return maximum PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Load the image
image = cv2.imread('lena.png')

# Resize the image to 256x256
image = cv2.resize(image, (256, 256))

# Convert to float32 for OpenCV DCT
image_float = np.float32(image)

# Apply optimized 2D-DCT using two 1D-DCTs to each channel
dct_optimized = np.zeros_like(image_float)
start_time = time.time()
for channel in range(3):
    dct_optimized[:, :, channel] = dct_2d(image_float[:, :, channel])
end_time = time.time()
print(f"Optimized 2D-DCT time (two 1D-DCTs): {end_time - start_time:.4f} seconds")

# Apply optimized 2D-IDCT using two 1D-IDCTs to each channel
reconstructed_image = np.zeros_like(image_float)
for channel in range(3):
    reconstructed_image[:, :, channel] = idct_2d(dct_optimized[:, :, channel])

# Clip to valid range [0, 255]
reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

# Calculate PSNR between the original and reconstructed image
psnr_brute = calculate_psnr(image, reconstructed_image)
print(f"PSNR: {psnr_brute:.2f} dB")

# Display results
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))
plt.title(f'Reconstructed - PSNR: {psnr_brute:.2f} dB')

plt.show()

# Compare the results between brute force and optimized
plt.imshow(np.log(np.abs(dct_optimized[:, :, 0]) + 1), cmap='gray')
plt.title('DCT (Optimized using two 1D-DCT)')
plt.show()

# start_time = time.time()
# dct_cv2 = cv2.dct(image_float)
# end_time = time.time()
# print(f"cv2 dct {end_time - start_time:.4f} seconds")
# idct_cv2 = cv2.idct(dct_cv2)
# idct_cv2 = np.clip(idct_cv2, 0, 255).astype(np.uint8)
# psnr_cv2 = calculate_psnr(image, idct_cv2)
# print(f"PSNR (cv2): {psnr_cv2:.2f} dB")