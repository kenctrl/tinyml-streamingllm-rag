import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys
import time

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
        
        self.vector_access_count = 0
        self.vector_access_history = []
        self.max_vectors = 100  # Maximum vectors to store
        self.clear_frequency = 10  # Clear LRU every 10 iteration
    
    def clear_lru_vector_store(self):
        """Clear the least recently used vector from the vector store"""
        if len(self.vector_access_history) > 0:
            print("Clearing LRU vector store...")
            lru_index = self.vector_access_history.pop(0)
            # FAISS doesn't support direct deletion, so we need to rebuild the index
            # excluding the LRU vector
            current_vectors = self.vector_store.docstore._dict
            filtered_texts = [
                doc.page_content for idx, doc in current_vectors.items() 
                if idx != str(lru_index)
            ]
            if filtered_texts:
                self.vector_store = FAISS.from_texts(filtered_texts, self.embeddings)
            else:
                self.vector_store = FAISS.from_texts([""], self.embeddings)
    
    def get_similarity_to_vector_store(self, text):
        """
        Get the score of the most similar context to the text from the vector store.
        """
        score = self.vector_store.similarity_search_with_score(text, k=1)
        return score[0][1]
        
    def store_evicted_tokens(self, evicted_text: str):            
        # If evicted_text is badly formatted, return
        if evicted_text is None or evicted_text == "" or not evicted_text or evicted_text[0] == "�":
            # print("\n\nBadly formatted evicted text.")
            return
        # If len(evicted_text) >= 3 and there are only a few unique characters other than punctuation, return
        if len(evicted_text) >= 3 and len(set(evicted_text.replace(".", "").replace(",", "").replace("!", "").replace("?", "").replace("-", "").replace("/", "").replace("\\", "").replace("\n", ""))) < 5:
            return
        # If evicted_text doesn't have a period, return
        # if "." not in evicted_text:
        #     return
        
        # Split evicted_text by newlines
        evicted_texts = evicted_text.split("\n")
        
        # Combine some consecutive evicted texts if any are fewer than 50 characters
        # combined_evicted_texts = []
        # for evicted_text in evicted_texts:
        #     if len(combined_evicted_texts) == 0:
        #         combined_evicted_texts.append(evicted_text)
        #     else:
        #         if len(combined_evicted_texts[-1]) + len(evicted_text) < 50: # TODO: mention this in the paper
        #             combined_evicted_texts[-1] += " " + evicted_text
        #         else:
        #             combined_evicted_texts.append(evicted_text)
        
        # Filter out evicted texts that are too short
        evicted_texts = [text for text in evicted_texts if len(text) > 10]
                    
        for evicted_text in evicted_texts:
            # Get the score of the most similar context to the evicted text from the vector store
            # score = self.get_similarity_to_vector_store(evicted_text)
            
            # If the score is less than 0.9, store the evicted text in the vector store
            if evicted_text != "" and evicted_text != " ":
                self.vector_store.add_texts([evicted_text])
            
        # print(f"\n\nStored {len(evicted_texts)} evicted tokens in vector store\n\n")

    def retrieve_relevant_context(self, text):
        results = self.vector_store.similarity_search(text, k=3)
        # return " ".join([doc.page_content for doc in results])
        if len(results) == 0 or (len(results) == 1 and results[0].page_content == ""):
            return ""
        out = "Top 3 contexts (may not be relevant):\n"
        # counter = 1
        for context in results:
            if context.page_content != " " and context.page_content != "" and context.page_content != "Top 3 contexts (may not be relevant):":
                out += f"{context.page_content}\n"
                # counter += 1
                
        return f"\n\n{out.strip()}"

@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    past_key_values = None
    rag_cache = RAGEnhancedKVCache()
    # prev_input_ids = None
    times = []
    tokens_per_second = []
    
    for idx, prompt in enumerate(prompts):
        start_time = time.time()
        if idx == 0:
            most_similar_context = ""
        else:
            most_similar_context = rag_cache.retrieve_relevant_context(prompt)
        
        prompt = f"\nUSER: {prompt}{most_similar_context}\n\nASSISTANT: "
        print(prompt, end="")
                
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids     
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            # Store evicted tokens before they're removed
            # evicted_raw_tokens is a list of tensors
            past_key_values, evicted_raw_tokens = kv_cache.evict_for_space(past_key_values, space_needed)
            
            if evicted_raw_tokens is not None and input_ids is not None:
                evicted_text = greedy_generate_text(model, tokenizer, input_ids, evicted_raw_tokens, max_gen_len=max_gen_len)
                # print("Evicted text: ", evicted_text)
                rag_cache.store_evicted_tokens(evicted_text)
            
        # prev_input_ids = input_ids

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )
        end_time = time.time()
        times.append(end_time - start_time)
        tokens_per_second.append(len(input_ids[0]) / (end_time - start_time))
        # print(f"\nTime taken: {end_time - start_time} seconds")
        
    return times, tokens_per_second


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    test_filepath = os.path.join(args.data_root, args.benchmark_file_name)
    print(f"Loading {args.benchmark_file_name} from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        print("Downloading benchmark file...")
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

    times, tokens_per_second = streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
    )
    
    print("All times: ", times)
    print(f"Average time per token: {sum(times) / len(times)}")
    print("All tokens per second: ", tokens_per_second)
    print(f"Average tokens per second: {sum(tokens_per_second) / len(tokens_per_second)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-7b-v1.3"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--benchmark_file_name", type=str, default="mt_bench.jsonl")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()

    main(args)
