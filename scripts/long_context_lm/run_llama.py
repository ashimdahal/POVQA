import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from headinfer.cache import OffloadedCache
from headinfer.mp import mp_headinfer, mp_simulate_decode

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2").to("cuda")

# Wrap the model with HeadInfer
# headinfer_model = HeadInferModel(model)

# Generate text with long context
input_text = "Once upon a time in a galaxy far, far away..."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")


with torch.inference_mode():

    # patch the model
    mp_headinfer(model)
    past_key_values = OffloadedCache()

    model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, num_logits_to_keep=1)
