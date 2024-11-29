MODE ?= server

CC = g++

CFLAGS = -fopenmp
LDFLAGS = 

ifeq ($(MODE), local)
    TARGET = main
    SRC = main.cpp
    CFLAGS += `pkg-config --cflags opencv4`
    LDFLAGS += `pkg-config --libs opencv4`
else ifeq ($(MODE), server)
    TARGET = main
    SRC = main.cpp
    LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
	export CPATH=/opt/spack/var/spack/environments/pp-env/.spack-env/view/include/opencv4:$CPATH
	export LD_LIBRARY_PATH := /opt/spack/var/spack/environments/pp-env/.spack-env/view/lib:$(LD_LIBRARY_PATH)
else
    $(error Unknown MODE: $(MODE))
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
