# Makefile for compiling test.cpp with OpenCV

CXX = g++
CXXFLAGS = -std=c++17 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`
TARGET = serial
SRC = serial.cpp

all: $(TARGET)

$(TARGET): $(SRC)
    $(CXX) -o $(TARGET) $(SRC) $(CXXFLAGS) $(LDFLAGS)

clean:
    rm -f $(TARGET)