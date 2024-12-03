# export CPATH=/opt/spack/var/spack/environments/pp-env/.spack-env/view/include/opencv4:$CPATH
# export LD_LIBRARY_PATH=/opt/spack/var/spack/environments/pp-env/.spack-env/view/lib:$LD_LIBRARY_PATH

MODE ?= server

SRC_DIR = src
BIN_DIR = bin

$(shell mkdir -p $(BIN_DIR))

NVCC = nvcc
NVCCFLAGS = -arch=sm_60 -Xcompiler -fopenmp -O3
LDFLAGS =

TARGET_OMP = $(BIN_DIR)/main
SRC_OMP = $(SRC_DIR)/main.cpp

TARGET_CUDA = $(BIN_DIR)/cuda
SRC_CUDA = $(SRC_DIR)/cuda.cpp $(SRC_DIR)/dct_cuda.cu

ifeq ($(MODE), local)
    NVCCFLAGS += `pkg-config --cflags opencv4`
    LDFLAGS += `pkg-config --libs opencv4`
else ifeq ($(MODE), server)
    LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
else
    $(error Unknown MODE: $(MODE))
endif

all: $(TARGET_OMP) $(TARGET_CUDA)

$(TARGET_OMP): $(SRC_OMP)
	$(NVCC) -o $@ $^ $(NVCCFLAGS) $(LDFLAGS)

$(TARGET_CUDA): $(SRC_CUDA)
	$(NVCC) -o $@ $^ $(NVCCFLAGS) $(LDFLAGS)

clean:
	-rm -rf $(BIN_DIR)
	-rm -f data/dct/*
	-rm -r data/reconstructed/*

.PHONY: all clean
