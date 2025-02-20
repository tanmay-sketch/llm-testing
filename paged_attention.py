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

        # We assume that the query is precomputed externally.
        # Therefore, we remove internal q_proj, k_proj, and v_proj layers.
        self.out_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)

    def forward(self, query, cache=None):
        """
        Args:
            query: A tensor of shape [batch_size, num_heads, 1, head_dim]
                   representing the query for the current token.
            cache: Optional QKVCache object. If None, a new cache is created.
        
        Returns:
            attn_output: The computed attention output with shape [batch_size, 1, num_heads * head_dim].
            cache: The updated QKVCache object.
        """
        batch_size, num_heads, q_len, head_dim = query.size()

        if cache is None:
            cache = QKVCache(batch_size, num_heads, head_dim, self.page_size, self.max_pages, self.device)

        # Retrieve the cached key-value pairs.
        key_cache, value_cache = cache.get_cache()

        # Compute attention scores using the provided query and cached keys.
        attn_scores = torch.matmul(query, key_cache.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Compute attention values.
        attn_values = torch.matmul(attn_scores, value_cache)  # Shape: (batch_size, num_heads, 1, head_dim)

        # Reshape the output to [batch_size, 1, num_heads * head_dim].
        attn_output = attn_values.transpose(1, 2).reshape(batch_size, 1, num_heads * head_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, cache