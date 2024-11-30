# Makefile for CUDA Version Only

MODE ?= server

# Compiler settings
NVCC = nvcc

NVCCFLAGS = -arch=sm_60 -Xcompiler -fopenmp -O3

LDFLAGS = 
NVCC_LDFLAGS =

# Target binary name
TARGET_CUDA = cuda

# Source files for the GPU (CUDA) version
SRC_CUDA = cuda.cpp dct.cu

# Libraries for OpenCV
ifeq ($(MODE), local)
	NVCCFLAGS += `pkg-config --cflags opencv4`
	NVCC_LDFLAGS += `pkg-config --libs opencv4`
else ifeq ($(MODE), server)
	NVCC_LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
	export CPATH=/opt/spack/var/spack/environments/pp-env/.spack-env/view/include/opencv4:$CPATH
	export LD_LIBRARY_PATH := /opt/spack/var/spack/environments/pp-env/.spack-env/view/lib:$(LD_LIBRARY_PATH)
else
	$(error Unknown MODE: $(MODE))
endif

# Compile the CUDA target
all: $(TARGET_CUDA)

$(TARGET_CUDA): $(SRC_CUDA)
	$(NVCC) -o $@ $^ $(NVCCFLAGS) $(NVCC_LDFLAGS)

# Clean command
clean:
	rm -f $(TARGET_CUDA)

.PHONY: all clean
