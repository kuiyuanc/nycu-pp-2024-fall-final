#!/bin/bash

OUT_DIR="logs"

if [ ! -d "$OUT_DIR" ]; then
    mkdir -p "$OUT_DIR"
fi

# serial
echo "Running serial..."
echo "./bin/main --method serial --num-threads 1 --full >> ./$OUT_DIR/serial.log"
./bin/main --method serial --num-threads 1 --full >> ./$OUT_DIR/serial.log
sleep 30

# cuda
echo "Running cuda..."
echo "./bin/main --method cuda --num-threads 1 --full >> ./$OUT_DIR/cuda.log"
./bin/main --method cuda --num-threads 1 --full >> ./$OUT_DIR/cuda.log
sleep 30

# pthread
for i in {6..1}; do
    echo "Running pthread with $i threads..."
    echo "./bin/main --method pthread --num-threads $i --full >> ./$OUT_DIR/pthread${i}.log"
    ./bin/main --method pthread --num-threads $i --full >> ./$OUT_DIR/pthread${i}.log
    sleep 30
done

# omp
for i in {6..1}; do
    echo "Running omp with $i threads..."
    echo "./bin/main --method omp --num-threads $i --full >> ./$OUT_DIR/omp${i}.log"
    ./bin/main --method omp --num-threads $i --full >> ./$OUT_DIR/omp${i}.log
    sleep 30
done

echo "Finished!"