import torch
from config import PAGE_SIZE, MAX_PAGES, DEVICE

class PhysicalBlock:
    def __init__(self):
        self.device = DEVICE
        self.max_pages = MAX_PAGES
        self.page_size = PAGE_SIZE

        # First column: page ID; second column: count of tokens added to that page.
        self.reference_counts = torch.zeros((self.max_pages, 2), dtype=torch.int16)
        self.reference_counts[:, 0] = torch.arange(1, self.max_pages + 1, dtype=torch.int16)

        # Use zeros to mark empty vectors.
        self.key_vector_empty = torch.zeros((768, 1))
        self.query_vector_empty = torch.zeros((768, 1))

        # Create a physical block tensor for all pages.
        # Each block will store a concatenated vector (key and value) of size 768*2.
        self.physical_block = torch.zeros(
            (self.max_pages, self.page_size, 768 * 2),
            dtype=torch.float16,
            device=self.device
        )

    def _getEmptyBlockInPage(self, page_index):
        """
        Returns the index of the first empty block in the page.
        A block is considered empty if all its elements are zero.

        Args:
            - page_index: The index of the page. Expected bounds: 0 to MAX_PAGES-1.
        
        Returns:
            - The index of the first empty block in the page, or -1 if no empty block is found.
        """
        # Iterate through blocks in the page.
        for i in range(self.page_size):
            block = self.physical_block[page_index][i]
            if torch.allclose(block, torch.zeros_like(block)):
                return i
        # If no empty block is found, return -1.
        return -1

    def _getEmptyPage(self):
        """
        Returns the index of the first empty page (i.e., a page with no tokens added).

        Returns:
            - The index of the first empty page, or -1 if all pages are full.
        """
        for i in range(self.max_pages):
            if self.reference_counts[i, 1] == 0:
                return i
        return -1


    def addKeyValue(self, page_index, key, value):
        """
        Adds new key-value vectors for a given token in the specified page.

        Args:
            page_index: The index of the page. Expected bounds: 0 to MAX_PAGES-1.
            key: The key vector of the token, expected to be of shape (768, 1).
            value: The value vector of the token, expected to be of shape (768, 1).

        Raises:
            ValueError: If the page is full or if the key/value shapes are incorrect.
        """
        # Validate the shapes of key and value.
        if key.shape != (768, 1) or value.shape != (768, 1):
            raise ValueError("Key and value must have shape (768, 1)")

        # Concatenate key and value to create a single vector of shape (1536, 1).
        concatenated_vector = torch.cat((key, value), dim=0)
        # Flatten to a 1D vector, matching the physical_block block shape (1536,).
        concatenated_vector = concatenated_vector.flatten()

        # Find the first empty block in the given page.
        empty_block_index = self._getEmptyBlockInPage(page_index)
        if empty_block_index == -1:
            raise ValueError("Page is full. Cannot add more tokens.")

        # Insert the concatenated vector into the found empty block.
        self.physical_block[page_index][empty_block_index] = concatenated_vector

        # Increment the token count for this page.
        self.reference_counts[page_index, 1] += 1
