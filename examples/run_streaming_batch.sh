#! /bin/bash

# run "CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py --benchmark_file_name=file --enable_streaming"for each of the files in the data/ directory
# output as file.txt in outputs/ directory
# handle crashes by skipping the file and continuing

for file in data-test/*; do
    echo "Running $file"
    CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py --benchmark_file_name=$file --enable_streaming > outputs/$(basename $file).txt 2> outputs/$(basename $file).err
    echo "Done with $file"
done
