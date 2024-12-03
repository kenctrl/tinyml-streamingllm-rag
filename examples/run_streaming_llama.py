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

@torch.no_grad()
def greedy_generate_text(model, tokenizer, input_ids, past_key_values, max_gen_len):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    text = ""
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
            new_text = " ".join(generated_text[pos:now])
            text += " " + new_text
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    new_text = " ".join(generated_text[pos:])
    text += " " + new_text
    return text.strip()

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    print("Generating text...")
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
    
    # def clear_lru_vector_store(self):
        
    
    def get_similarity_to_vector_store(self, text):
        """
        Get the score of the most similar context to the text from the vector store.
        """
        score = self.vector_store.similarity_search_with_score(text, k=1)
        return score[0][1]
        
    def store_evicted_tokens(self, evicted_text: str):            
        # If evicted_text is badly formatted, return
        if "�" in evicted_text:
            print("\n\nBadly formatted evicted text: ", evicted_text, "\n\n")
            return
        
        # Get the score of the most similar context to the evicted text from the vector store
        score = self.get_similarity_to_vector_store(evicted_text)
        
        # If the score is less than 0.5, store the evicted text in the vector store
        if score < 0.8:
            self.vector_store.add_texts([evicted_text])
            print("\n\nStored evicted tokens in vector store\n\n")

    def retrieve_relevant_context(self, text):
        results = self.vector_store.similarity_search(text, k=2)
        return " ".join([doc.page_content for doc in results])

@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    past_key_values = None
    rag_cache = RAGEnhancedKVCache()
    prev_input_ids = None
    
    for idx, prompt in enumerate(prompts):
        most_similar_context = rag_cache.retrieve_relevant_context(prompt)
        
        print("Printing prompt...")
        prompt = f"USER: {prompt}\n\nContext from previous conversations (NOTE: this may not be relevant to the current conversation): {most_similar_context}\n\nASSISTANT: "
        print("\n" + prompt, end="")
                
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids     
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            # Store evicted tokens before they're removed
            # evicted_raw_tokens is a list of tensors
            past_key_values, evicted_raw_tokens = kv_cache.evict_for_space(past_key_values, space_needed)
            
            if evicted_raw_tokens is not None and prev_input_ids is not None:
                evicted_text = greedy_generate_text(model, tokenizer, prev_input_ids, evicted_raw_tokens, max_gen_len=max_gen_len)
                # print("Evicted text: ", evicted_text)
                rag_cache.store_evicted_tokens(evicted_text)
            
        prev_input_ids = input_ids

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
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
