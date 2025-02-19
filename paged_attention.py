import torch.nn as nn

class PagedAttention(nn.Module):
    def __init__(self, num_heads, head_dim, paged_attention):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.paged_attention = paged_attention
    