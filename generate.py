import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpt2_attention import GPT2PagedAttention

PAGE_SIZE = 64
MAX_PAGES = 16
DEVICE = torch.device("mps")  # Using MPS

# Load pre-trained GPT-2 and move it to MPS.
model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Patch transformer blocks with GPT2PagedAttention.
for i, block in enumerate(model.transformer.h):
    paged_attn = GPT2PagedAttention(
        config=model.config,
        page_size=PAGE_SIZE,
        max_pages=MAX_PAGES,
        device=DEVICE
    ).to(DEVICE)
    block.attn = paged_attn
    print(f"Replaced block {i} attention with GPT2PagedAttention")

prompt = "If I am a wolf, then you are a"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

# Generate text with use_cache=True to invoke the paged attention mechanism.
generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=True, use_cache=True)
print(tokenizer.decode(generated_ids[0]))