# Use an official Ubuntu image as a base
FROM ubuntu:20.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    pkg-config \
    wget && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Fix Makefile tabs if needed (optional step for safety)
RUN sed -i 's/^    /\t/' Makefile

# Build the project
RUN make

# Run the program during container startup
CMD ["./serial"]
