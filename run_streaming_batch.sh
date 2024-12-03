#! /bin/bash

# run "CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py --benchmark_file_name=file --enable_streaming"for each of the files in the data/ directory
# output as file.txt in outputs/ directory
# handle crashes by skipping the file and continuing

mkdir -p outputs
for file in data-test/*; do
    echo "Running $file"
    base=$(basename $file)
    CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py --benchmark_file_name=$file --enable_streaming > outputs/$base-streaming-rag.txt 2> outputs/$base-streaming-rag.err
    CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama_original.py --benchmark_file_name=$file --enable_streaming > outputs/$base-streaming-original.txt 2> outputs/$base-streaming-original.err
    CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama_original.py --benchmark_file_name=$file > outputs/$base-no-streaming-original.txt 2> outputs/$base-no-streaming-original.err
    echo "Done with $file"
done
