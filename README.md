# Implementation of Parallelized DCT & IDCT

NYCU Parallel Programming 2024 Final Project
Author: Yu-Shiang Tsai, Kui-Yuan Chen, Peng-Hsuan Huang

## Introduction

Our works unveiled the weakness and strength of the three programming models, Pthread, OpenMP, and CUDA. By adopting different solutions depending on the circumstance, we could leverage parallelism to boost DCT and IDCT.


Our evalution platform:
- OS: Ubuntu 24.04 LTS
- Processor: AMD Ryzen 3 3500x 6-Core 
- GPU: Nvidia RTX-4060-TI 
- RAM: DDR4-3200 32G


## Envirorment Setup
1. Install `nvcc` and `OpenCV`
    ```
    sudo apt install nvidia-cuda-toolkit -y
    sudo apt install libopencv-dev -y
    ```
2. Check installation
    ```
    nvcc --version
    pkg-config --modversion opencv4
    ```
    Output:
    ```
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2023 NVIDIA Corporation
    Built on Fri_Jan__6_16:45:21_PST_2023
    Cuda compilation tools, release 12.0, V12.0.140
    Build cuda_12.0.r12.0/compiler.32267302_0
    
    4.6.0
    ```
    
3. Extract the project
    The File structure should look like:
    - `root`
        - `data`
            - `original`
                - `0-99.png`
        - `src`
            - `main.cpp`
            - `dct_cuda.cu`
            - `dct_cuda.hpp`
            - `dct_omp.hpp`
            - `dct_pthread.hpp`
            - `dct_serial.hpp`
            - `experiment.hpp`
        - `Makefile`
        - `run_full.sh`
        - `run.sh`
4. Build
- Bulid in local environment
    ```
    make
    ```
- Bulid in PP-Server
    ```
    export LD_LIBRARY_PATH=/opt/spack/var/spack/environments/pp-env/.spack-env/view/lib:$LD_LIBRARY_PATH >> ~/.bashrc
    export CPATH=/opt/spack/var/spack/environments/pp-env/.spack-env/view/include/opencv4:$CPATH >> ~/.bashrc
    source ~/.bashrc
    make MODE=server
    ```

# How to run the experiments



Run single image only
```
bash run.sh
```

Run full-datasets(100 images)
```
bash run_full.sh
```
Notice: If you want to run full dataset, please make sure there are **16GB** free memory available.(or try to remove some images `data/original/*.png`)




The results will be in `./logs/*.log`

Sample result:
```
================================================================================
Intermediate data are discarded
Loading data from data/original
	Loading 0.png only
	Image shape: (512, 512)
Testing with following parameters:
	Method: Serial
	Using 1 threads
	Number of tests = 30
PSNR validation acceptance tolerance = 1.0e-05
================================================================================
DCT:
	Mean:			 0.01285 s
	95% CI:			[0.01265, 0.01304] s
IDCT:
	Mean:			 0.01205 s
	95% CI:			[0.01185, 0.01226] s
PSNR validation:	Pass
PSNR mean:		100.00000
================================================================================
```