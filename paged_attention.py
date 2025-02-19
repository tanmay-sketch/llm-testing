import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from qkv import QKVCache

class PagedAttention(nn.Module):
    def __init__(self, num_heads, head_dim, page_size, max_pages, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )

        # Linear projection layers for queries, keys, and values
        self.q_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)

        # Paged key-value cache will be instantiated per batch
        self.page_size = page_size
        self.max_pages = max_pages

    def forward(self,x,cache=None):
        """
        Args:
        - x: A tensor of shape [batch_size, seq_len, embed_dim]
        - cache: Optional QKV cache object, if None, a new cache will be created
        """

        batch_size, seq_len, embed_dim = x.size()

        if cache is None:
            cache = QKVCache(batch_size, self.num_heads, self.head_dim, self.page_size, self.max_pages, self.device)

        # Compute the linear projections for queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cache.add(k, v)

        # Retrieve the cached key-value pairs
        key_cache, value_cache = cache.get_cache()

        # compute the query for the current token
        curr_q = q[:, -1:, :, :]

        # Compute the attention scores
        attn_scores = torch.matmul(curr_q, key_cache.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Compute attention values
        attn_values = torch.matmul(attn_scores, value_cache)

        attn_output = attn_values.transpose(1, 2).reshape(batch_size, 1, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, cache



    