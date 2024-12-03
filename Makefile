# export CPATH=/opt/spack/var/spack/environments/pp-env/.spack-env/view/include/opencv4:$CPATH
# export LD_LIBRARY_PATH=/opt/spack/var/spack/environments/pp-env/.spack-env/view/lib:$LD_LIBRARY_PATH

MODE ?= server

SRC_DIR = src
BIN_DIR = bin

$(shell mkdir -p $(BIN_DIR))

NVCC = nvcc
NVCCFLAGS = -arch=sm_60 -Xcompiler -fopenmp -O3
LDFLAGS =

ifeq ($(MODE), local)
    NVCCFLAGS += `pkg-config --cflags opencv4`
    LDFLAGS += `pkg-config --libs opencv4`
else ifeq ($(MODE), server)
    LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
else
    $(error Unknown MODE: $(MODE))
endif

all:

clean:
	-rm -rf $(BIN_DIR)
	-rm -f data/dct/*
	-rm -r data/reconstructed/*

.PHONY: all clean
