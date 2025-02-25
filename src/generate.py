import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpt2_attention import GPT2PagedAttention  # Your custom attention class
from config import DEVICE, PAGE_SIZE, MAX_PAGES

prompt = "If I am a wolf, then your are ..."

# ----------------------
# Regular GPT-2 Output
# ----------------------
model= AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

# Generate with the default GPT2Attention mechanism
generated_ids_regular = model.generate(**inputs, max_length=50, do_sample=True)
output_text_regular = tokenizer.decode(generated_ids_regular[0])
print(model.config)