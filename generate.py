import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpt2_attention import GPT2PagedAttention  # Your custom attention class

PAGE_SIZE = 64
MAX_PAGES = 16
DEVICE = torch.device("mps")  # Using MPS

prompt = "If I am a wolf, then you are a"

# ----------------------
# Regular GPT-2 Output
# ----------------------
model_regular = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
# Generate with the default GPT2Attention mechanism
generated_ids_regular = model_regular.generate(**inputs, max_new_tokens=50, do_sample=True)
output_text_regular = tokenizer.decode(generated_ids_regular[0]).strip()

# ----------------------
# GPT2PagedAttention Output
# ----------------------
model_paged = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)

# Patch each transformer block to use GPT2PagedAttention.
for i, block in enumerate(model_paged.transformer.h):
    paged_attn = GPT2PagedAttention(
        config=model_paged.config,
        page_size=PAGE_SIZE,
        max_pages=MAX_PAGES,
        device=DEVICE
    ).to(DEVICE)
    block.attn = paged_attn
    print(f"Replaced block {i} attention with GPT2PagedAttention")

# Use use_cache=True so that your custom caching is activated.
generated_ids_paged = model_paged.generate(**inputs, max_new_tokens=50, do_sample=True, use_cache=True)
output_text_paged = tokenizer.decode(generated_ids_paged[0]).strip()

print("Regular GPT-2 Output:")
print(output_text_regular)
print("\nGPT2PagedAttention Output:")
print(output_text_paged)