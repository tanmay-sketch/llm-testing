import torch 

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
PAGE_SIZE = 16
MAX_PAGES = 64