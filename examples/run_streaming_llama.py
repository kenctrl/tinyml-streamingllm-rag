import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

api_key = os.environ.get("OPENAI_API_KEY")

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, rag_cache, max_gen_len):
    # RAG Cache contains raw evicted tokens from previous generations.
    # Perform retrieval on the evicted tokens and add the retrieved tokens to the vector store.
    most_similar_tokens = rag_cache.retrieve_relevant_context(input_ids)
    print("Most similar tokens: ", most_similar_tokens)
    
    # Append the most similar tokens to the input ids
    input_ids = torch.cat([input_ids, most_similar_tokens], dim=-1)
    
    print("Input ids: ", input_ids)
    
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values


class RAGEnhancedKVCache:
    def __init__(self, embedding_model="sentence-transformers/all-mpnet-base-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = FAISS.from_texts([""], self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.tokenizer = None  # Will be set during initialization
        
    def store_evicted_tokens(self, evicted_data, tokenizer):
        if self.tokenizer is None:
            self.tokenizer = tokenizer
            
        # Store raw evicted tokens in vector store
        for k, v in evicted_data:
            self.vector_store.add_texts(tokenizer.decode(k.cpu().tolist(), skip_special_tokens=True))
            
        print("Stored evicted tokens in vector store")

    def retrieve_relevant_context(self, input_ids):
        results = self.vector_store.similarity_search(input_ids, k=3)
        return " ".join([doc.page_content for doc in results])

@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=200):
    past_key_values = None
    rag_cache = RAGEnhancedKVCache()
    
    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        
        # Get relevant past context
        if idx > 0:
            relevant_context = rag_cache.retrieve_relevant_context(prompt)
            prompt = f"Previous relevant context: {relevant_context}\n\nCurrent query: {prompt}"
            print("PROMPT WITH CONTEXT: ", prompt)
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            # Store evicted tokens before they're removed
            past_key_values, evicted_tokens = kv_cache.evict_for_space(past_key_values, space_needed)
            
            if evicted_tokens is not None:
                rag_cache.store_evicted_tokens(evicted_tokens, tokenizer)

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, rag_cache, max_gen_len=max_gen_len
        )


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()

    main(args)
