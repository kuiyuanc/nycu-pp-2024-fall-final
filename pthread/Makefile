# Makefile for compiling test.cpp with OpenCV

CXX = g++
CXXFLAGS = -std=c++17 -lpthread `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`
TARGET = pthread
SRC = pthread.cpp

all: $(TARGET)

$(TARGET): $(SRC)
    $(CXX) -o $(TARGET) $(SRC) $(CXXFLAGS) $(LDFLAGS)

clean:
    rm -f $(TARGET)