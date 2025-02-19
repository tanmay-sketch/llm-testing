import torch

class QKVCache:
    """
    A class to store the key-value pairs for the QKV attention mechanism. 
    
    The cache uses pre-allocated tensors to store the key-value pairs 
    """
    def __init__(self, batch_size, num_heads, head_dim, page_size, max_pages, device=None):
        """
        Args: 
        - batch_size: Number of sequences in the batch
        - num_heads: Number of attention heads
        - head_dim: Dimension of each attention head
        - page_size: Number of tokens stored in each page
        - max_pages: Maximum number of pages to store
        - device: Device to store the cache tensors (default: cuda if available, mps if available, else cpu)
        """

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )

        # Use lists to store pages of keys and values
        self.key_pages = []
        self.value_pages = []

        self.current_length = 0 # Number of tokens stored in the cache
        self.current_page_index = -1 
        self.current_offset = 0 # Offset in the current page

        self._allocate_new_page()

    def _allocate_new_page(self):
        """
        Private method to allocate a new page of key-value pairs
        """
        # Allocate a new page (a tensor of shape [batch_size, num_heads, page_size, head_dim])
        key_page = torch.empty(self.batch_size, self.num_heads, self.page_size, self.head_dim, device=self.device)
        value_page = torch.empty(self.batch_size, self.num_heads, self.page_size, self.head_dim, device=self.device)
        self.key_pages.append(key_page)
        self.value_pages.append(value_page)
        self.current_page_index += len(self.key_pages) - 1
        self.current_offset = 0

    def add(self, key, value):
        """
        Adds a key-value pair to the cache. Both key and value should have the same shape.

        Args:
        - key: A tensor of shape [batch_size, num_heads, seq_len, head_dim]. seq_len is the length of new tokens to be added
        - value: A tensor of shape [batch_size, num_heads, seq_len, head_dim]. seq_len is the length of new tokens to be added

        Returns:
        - None

        Raises:
        - AssertionError: If the shape of key and value tensors do not match
        - ValueError: If the cache is full
        """

        assert key.shape == value.shape, "Key and value tensors must have the same shape"

        num_tokens = key.size(2)
        token_offset = 0

        #Process the new tokens in chunks so that they fit in the current page
        while token_offset < num_tokens:
            remaining_space = self.page_size - self.current_offset
            token_to_copy = min(remaining_space, num_tokens - token_offset)

            current_key_page = self.key_pages[self.current_page_index]
            current_value_page = self.value_pages[self.current_page_index]

            # Copy tokens into the current page at the right offset
            current_key_page[:, :, self.current_offset:self.current_offset + token_to_copy, :] = key[:, :, token_offset:token_offset + token_to_copy, :]
            current_value_page[:, :, self.current_offset:self.current_offset + token_to_copy, :] = value[:, :, token_offset:token_offset + token_to_copy, :]
            
            self.current_offset += token_to_copy
            self.current_length += token_to_copy
            token_offset += token_to_copy

            if self.current_offset == self.page_size:
                if len(self.key_pages) < self.max_pages:
                    self._allocate_new_page()
                else:
                    raise ValueError("Cache is full")
                
    def get_cache(self):
        """
        Retrieves the full key and value caches as a tuple of contiguous tensors
        The pages are concatenated along the seq_len (sequence) dimension

        Returns:
        - A tuple of two tensors: (key_cache, value_cache) 
        """
        key_list = []
        value_list = []

        # All pages except the last one should be full
        for i in range(len(self.key_pages) - 1):
            key_list.append(self.key_pages[i])
            value_list.append(self.value_pages[i])

        # The last page should only have tokens until the current offset
        key_list.append(self.key_pages[-1][:, :, :self.current_offset, :])
        value_list.append(self.value_pages[-1][:, :, :self.current_offset, :])

        key_cache = torch.cat(key_list, dim=2)
        value_cache = torch.cat(value_list, dim=2)
        return (key_cache, value_cache)
