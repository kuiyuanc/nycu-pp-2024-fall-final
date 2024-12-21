# cd nycu-pp-2024-fall-final; export CPATH=/opt/spack/var/spack/environments/pp-env/.spack-env/view/include/opencv4:$CPATH; export LD_LIBRARY_PATH=/opt/spack/var/spack/environments/pp-env/.spack-env/view/lib:$LD_LIBRARY_PATH

MODE ?= server

SRC_DIR = src
LIB_DIR = lib
BIN_DIR = bin

$(shell mkdir -p $(BIN_DIR))

NVCC = nvcc
NVCCFLAGS = -arch=sm_60 -Xcompiler -fopenmp -O3 
LDFLAGS =

TARGET = $(BIN_DIR)/main
SRC = $(LIB_DIR)/util.cpp $(SRC_DIR)/dct_cuda.cu $(SRC_DIR)/main.cpp

ifeq ($(MODE), local)
    NVCCFLAGS += `pkg-config --cflags opencv4`
    LDFLAGS += `pkg-config --libs opencv4` -ltbb -diag-suppress=611
else ifeq ($(MODE), server)
    LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
else
    $(error Unknown MODE: $(MODE))
endif


all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $@ $^ $(NVCCFLAGS) $(LDFLAGS)

clean:
	-rm -rf $(BIN_DIR)
	-rm -f data/dct/*/*
	-rm -f data/reconstructed/*/*

.PHONY: all clean
