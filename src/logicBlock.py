import torch
from config import PAGE_SIZE, MAX_PAGES, DEVICE
from physicalBlock import PhysicalBlock

class LogicBlock:
    def __init__(self):
        self.device = DEVICE
        self.max_pages = MAX_PAGES
        self.page_size = PAGE_SIZE
        self.query_vector_size = 768
        self.key_vector_size = 768
        self.logical_block = self.logical_block_block = torch.zeros(
            (self.max_pages, self.page_size, 768 * 2),
            dtype=torch.float16,
            device=self.device
        )
        self.current_page = 0
        
    

    def _generateHashForBlock(self, page_index, block_index):
        """
        This is a private function that generates the hash for a block based on the 
        prefix tokens and the block tokens.

        In other words, it generates a hash for the block based on the tokens that are
        present in the block and the sequence of tokens that appeared before it.

        Args:
            - page_index: The index of the page.
            - block_index: The index of the block in the page.
        Returns:
            - The hash for the block.
        """
        if page_index < 0 or page_index >= self.max_pages:
            raise IndexError("Page index out of bounds.")
        
        if block_index < 0 or block_index >= self.page_size:
            raise IndexError("Block index out of bounds.")
        block = self.logical_block[page_index][block_index]
    
        if block_index == 0:
            data = block.view(-1)
        else:
            prefix_tokens = self.logical_block[page_index][:block_index].view(-1)
            data = torch.cat((prefix_tokens, block.view(-1)))
        
        numel = data.numel()
        weights = torch.arange(1, numel + 1, device=data.device, dtype=data.dtype)

        hash1 = torch.sum(data * weights)
        weights2 = weights * torch.pi
        hash2 = torch.sum(data * weights2)
        hash_value = (hash1 + hash2 + 31)
        return hash_value


    def _getfirstEmptyBlockInPage(self):
        """
        This is a private function that returns the index of the first empty block in a page.
        """

        for i in range(self.page_size):
            block = self.logical_block[self.current_page][i]
            if torch.allclose(block, torch.zeros_like(block)):
                return i
        return -1

    def getPage(self, page_index):
        """
        The get page method retrieves a page from the logical block.
        """
        if page_index < 0 or page_index >= self.max_pages:
            raise IndexError("Page index out of bounds.")
        return self.logical_block[page_index]
    
    def getPageSize(self,page_index):
        """
        The get page size method retrieves the size of a page.
        """
        for i in range(self.page_size):
            block = self.logical_block[page_index][i]
            if torch.allclose(block, torch.zeros_like(block)):
                return i + 1
        return self.page_size 

    def addToken(self, key_vector, value_vector):
        """
        Add a token's concatenated key and value vectors to the current page.
        When the page (row) is full, start a new page.
        """

        concatenated_vector = torch.cat((key_vector, value_vector), dim=0).view(-1)
        index = self._getfirstEmptyBlockInPage()

        # If the current page is full (no empty block found), move to next page.
        if index == -1:
            self.current_page += 1
            if self.current_page >= self.max_pages:
                raise ValueError("Logical block is full.")
            index = self._getfirstEmptyBlockInPage()

        # Place the token in the first empty block of the current page.
        self.logical_block[self.current_page][index] = concatenated_vector
 
