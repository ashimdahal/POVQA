#!/usr/bin/env python
# qwen25vl_headinfer_demo.py
# Tested with transformers-4.40, torch-2.2, headinfer patch from our chat.

import time 
import torch
import types
import contextlib

from pathlib import Path
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from headinfer.mp    import mp_headinfer, mp_simulate_decode
from headinfer.cache import OffloadedCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"      # 7B also works

# ──────────────────────────────────────────────────────────────────────
# 1.  load tokenizer & model
# ──────────────────────────────────────────────────────────────────────
tok   = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.pad_token or tok.eos_token          # silence warnings

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype         = DTYPE,
    low_cpu_mem_usage   = True,
    attn_implementation = "flash_attention_2",
    trust_remote_code   = True,
).to(DEVICE).eval()

# Apply HeadInfer patch
mp_headinfer(model)

current_dir = Path(__file__).parent
book = open(current_dir / "alice.txt").read()

question = (
    "\n\n### User: Summarise Chapter 1 in one paragraph.\n### Assistant:"
)
prompt_text = book + question

input_ids = tok(prompt_text, return_tensors="pt").input_ids.to(DEVICE)
prompt_len = input_ids.shape[1]
print("prompt tokens:", prompt_len)

@contextlib.contextmanager
def disable_lm_head(model):
    real_head = model.lm_head
    model.lm_head = torch.nn.Identity()
    try:
        yield
    finally:
        model.lm_head = real_head

t0 = time.time()
# ──────────────────────────────────────────────────────────────────────
# 3.  PREFILL  (stream KV to CPU with OffloadedCache)
# ──────────────────────────────────────────────────────────────────────
kv_cache = OffloadedCache()

with torch.inference_mode(), disable_lm_head(model):
    _ = model(
        input_ids       = input_ids,          # 230 k-token prompt
        attention_mask  = None,
        pixel_values    = None, pixel_values_videos=None,
        past_key_values = kv_cache,
        use_cache       = True,
        return_dict     = False,              # tiny extra saving
    )

torch.cuda.synchronize() if DEVICE == "cuda" else None
print(f"prefill done in {time.time()-t0:.1f}s  –  cache layers: {len(kv_cache.key_cache)}")

# ──────────────────────────────────────────────────────────────────────
# 4.  GREEDY DECODING LOOP  (no .generate())
#     *one* token at a time, keeps GPU RAM flat
# ──────────────────────────────────────────────────────────────────────
new_tokens = 32
next_input = input_ids[:, -1:]                          # last prompt token  (B,1)
cache_pos  = torch.tensor([prompt_len-1],               # its index in stream
                          device=DEVICE, dtype=torch.long)

generated = [next_input.squeeze(0)]                     # store new tokens

for step in range(new_tokens):
    with torch.inference_mode():
        logits = model(
            input_ids       = next_input,
            attention_mask  = torch.ones_like(next_input),
            past_key_values = kv_cache,
            cache_position  = cache_pos,
            pixel_values    = None, pixel_values_videos=None,
            use_cache       = True,
            return_dict     = False,
        )[0]                                            # (B,1,V)

    next_token = logits[:, -1].argmax(-1, keepdim=True) # greedy
    generated.append(next_token.squeeze(0))

    # advance
    next_input = next_token
    cache_pos += 1

# ──────────────────────────────────────────────────────────────────────
# 5.  show result
# ──────────────────────────────────────────────────────────────────────
decoded = tok.decode(torch.cat(generated).tolist(), skip_special_tokens=True)
print("\nassistant:", decoded)


