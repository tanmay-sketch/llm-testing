import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps"

model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print(model.config)

prompt = "If I am a wolf, then you are a"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])
