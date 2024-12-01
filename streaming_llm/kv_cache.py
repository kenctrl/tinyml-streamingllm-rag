from llama_index.core import Document
from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.indices.vector_store import VectorStoreIndex
import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]


class EnhancedKVCache(StartRecentKVCache):
    def __init__(self, start_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2):
        super().__init__(start_size, recent_size, k_seq_dim, v_seq_dim)
        print(f"EnhancedKVCache: {start_size}, {recent_size}")
        # Initialize vector store for evicted tokens
        self.vector_store = SimpleVectorStore()
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        self.evicted_chunks = []
        
    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
            
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        # Store evicted tokens
        eviction_start = self.start_size
        eviction_end = seq_len - self.recent_size + num_coming
        
        for k, v in past_key_values:
            # Extract evicted tokens' KV pairs
            evicted_k = self.k_slice(k, eviction_start, eviction_end)
            evicted_v = self.v_slice(v, eviction_start, eviction_end)
            
            # Convert to embeddings and store in vector DB
            # Note: You'll need to implement embedding conversion based on your model
            embedding = self._convert_kv_to_embedding(evicted_k, evicted_v)
            doc = Document(text=f"chunk_{len(self.evicted_chunks)}", embedding=embedding)
            self.index.insert(doc)
            self.evicted_chunks.append((evicted_k, evicted_v))

        # Perform original eviction
        return super().evict_for_space(past_key_values, num_coming)

    def retrieve_relevant_chunks(self, current_context, top_k=3):
        """Retrieve most relevant evicted chunks based on current context"""
        # Convert current context to query embedding
        query_embedding = self._convert_context_to_embedding(current_context)
        
        # Query vector store
        results = self.index.query(
            query_embedding,
            top_k=top_k,
            mode="embedding"
        )
        
        # Return relevant KV pairs
        relevant_chunks = [self.evicted_chunks[int(r.text.split('_')[1])] 
                         for r in results]
        return relevant_chunks

    def _convert_kv_to_embedding(self, k, v):
        """Convert KV pairs to embeddings for storage
        This needs to be implemented based on your specific model architecture"""
        # Placeholder implementation
        combined = torch.cat([k.mean(dim=[0,1]), v.mean(dim=[0,1])], dim=0)
        return combined.detach().cpu().numpy()

    def _convert_context_to_embedding(self, context):
        """Convert current context to embedding for querying
        This needs to be implemented based on your specific model architecture"""
        # Placeholder implementation
        return context.mean(dim=[0,1]).detach().cpu().numpy()
