# StreamingRAG: Efficient Long-Context Inference with LLMs
[[slides](slides.pdf)][[video](https://youtu.be/sTw1TQ400pQ)]

## Abstract
In scenarios where LLMs are used for long-context tasks, one major challenge is the computation and memory costs that occur due to the capacities of key value caches. A potential solution that aims to address this issue is StreamingLLM, which accomplishes the computation and memory costs by way of preserving initial tokens to leverage attention sinks. However, a bottleneck that StreamingLLM faces is the case where tokens are evicted from the sliding window. To help mitigate this downside, we propose \textit{StreamingRAG}. We utilize Retrieval-Augmented Generation (RAG) to store the evicted tokens from StreamingLLM into a vector database and then selectively inject them into prompts. StreamingRAG effectively extends the context window length infinitely without sacrificing efficiency. StreamingRAG showcases an improved accuracy in tasks that require a large context window, such as document question answering, while maintaining a 22x speedup over the Sliding Window w/ Recomputation.

## Usage
Our contributions can be found in `examples/`.

### Environment Setup
The setup is the same as that for [StreamingLLM](https://github.com/mit-han-lab/streaming-llm), with one caveat. Since we use GPT-4o-mini-as-a-judge to score LLM outputs, you'll need to paste an OpenAI key in `produce_statistics.py` to collect results.

```bash
conda create -yn streaming python=3.8
conda activate streaming

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
```

### Run StreamingRAG Chatbot on MT-Bench

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py --enable_streaming
```

### Push-the-button commands to run all benchmarks
```bash
./run_streaming_batch
./produce_statistics.py
```

This will run StreamingRAG, StreamingLLM, and Dense Attentions on all benchmarks specified in `run_streaming_batch.sh`. Outputs can be found in `outputs/`. Then, it will run GPT-4o-mini-as-a-judge to score LLM outputs, as described in the paper.

Modify `run_streaming_batch.sh` as needed to include or exclude benchmarks from the `data/**` directories. Also, make sure to modify the outputs directory in `produce_statistics.py` to correctly get the benchmark data that was generated. Visualization scripts can be found in `stats/`.
