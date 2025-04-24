import torch
import transformers
from transformers import AutoTokenizer, AutoConfig, Qwen2_5_VLForConditionalGeneration
from PIL import Image # May be needed by model init
import os
import time
import traceback

# --- Import from the installed headinfer library ---
try:
    from headinfer.mp import mp_headinfer
    # Assuming OffloadedCache is in headinfer.cache
    from headinfer.cache import OffloadedCache
    print("Successfully imported mp_headinfer and OffloadedCache from headinfer library.")
except ImportError as e:
    print(f"Error: Could not import from headinfer library: {e}")
    print("Please ensure the headinfer library (with mp.py and cache.py) is installed correctly in your Python environment.")
    exit(1)
# --- No redefinition of library functions here ---

# --- Test Setup ---
# Using the specified Qwen2.5 VL model identifier
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load Tokenizer ---
print(f"Loading tokenizer for {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Ensure tokenizer has a padding token if it doesn't (use EOS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer pad_token to eos_token ({tokenizer.pad_token})")
    print("Tokenizer loaded.")
except Exception as e: print(f"Error loading tokenizer: {e}"); exit(1)

# --- Load Full Model ---
# Changed from _from_config to from_pretrained to load weights
print(f"Loading full model {model_name} (this may take time and VRAM)...")
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        # low_cpu_mem_usage=True, # Can help if CPU RAM is limited
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager" # Use FA2 if available
    ).to(device)
    model.eval() # Set to evaluation mode
    print("Full model loaded.")
except Exception as e: print(f"Error loading full model: {e}"); traceback.print_exc(); exit(1)

# --- Apply HeadInfer Patch ---
print("\n--- Applying HeadInfer Patch ---")
try:
    mp_headinfer(model) # Apply patch from the imported library function
except Exception as e: print(f"Error applying patch: {e}"); traceback.print_exc(); exit(1)

# --- Prepare Input Data ---
input_file = "alice.txt"
dummy_file = "dummy_alice.txt"
num_dummy_tokens = 100000 # Target ~100k tokens for dummy file

if os.path.exists(input_file):
    print(f"Reading text from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f: text = f.read()
        print(f"Read {len(text)} characters.")
    except Exception as e:
        print(f"Error reading {input_file}: {e}"); exit(1)
else:
    print(f"{input_file} not found. Creating dummy file {dummy_file} with ~{num_dummy_tokens} tokens...")
    sentence = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?' "
    try:
        approx_tokens_per_sentence = len(tokenizer.encode(sentence))
        repetitions = max(1, num_dummy_tokens // approx_tokens_per_sentence) if approx_tokens_per_sentence > 0 else num_dummy_tokens
        text = (sentence + "\n") * repetitions
        with open(dummy_file, 'w', encoding='utf-8') as f: f.write(text)
        print(f"Created {dummy_file} with {repetitions} repetitions.")
        input_file = dummy_file
    except Exception as e:
        print(f"Error creating dummy file: {e}"); exit(1)

print(f"Tokenizing text from {input_file}...")
try:
    # Tokenize the potentially long text
    # Set padding side if needed, though for single sequence prefill it might not matter
    tokenizer.padding_side = "left"
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device) # Get mask as well

    # Optional: Limit sequence length if needed
    max_seq_len_test = 100 * 1024
    if input_ids.shape[1] > max_seq_len_test:
        print(f"Warning: Input sequence length ({input_ids.shape[1]}) > {max_seq_len_test}. Truncating.")
        input_ids = input_ids[:, -max_seq_len_test:] # Keep the end if padding left
        attention_mask = attention_mask[:, -max_seq_len_test:]

    print(f"Using input_ids shape: {input_ids.shape}")
except Exception as e: print(f"Error tokenizing text: {e}"); exit(1)
del text # Free memory

# --- Initialize OffloadedCache ---
past_key_values = OffloadedCache()

# --- Run Forward Pass (Test Execution with Long Context) ---
print("\n--- Running Forward Pass (Patched + OffloadedCache + Long Text) ---")
# Removed VRAM measurement setup
try:
    with torch.inference_mode():
        start_time = time.time()
        print("Executing model forward pass...")
        # Call the main model's forward pass with the long input
        # The model will internally handle chunking or processing based on its limits
        # if needed, but here we pass the whole sequence for the prefill test.
        # We pass pixel_values=None as we are only testing text path patching.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask, # Pass the 2D mask
            pixel_values=None,
            past_key_values=past_key_values, # Pass the cache object
            use_cache=True # Enable caching
            )
        torch.cuda.synchronize(device) if device == "cuda" else None # Sync for timing
        end_time = time.time()
        print(f"Forward pass successful. Logits shape: {outputs.logits.shape}")
        print(f"Execution time: {end_time - start_time:.3f} seconds")
        # Check if cache was populated
        if len(past_key_values.key_cache) > 0:
            print(f"OffloadedCache was populated with {len(past_key_values.key_cache)} layers.")
        else:
            print("Warning: OffloadedCache was not populated.")

    print("\n[SUCCESS] HeadInfer patch applied and forward pass with long text executed without errors.")

except Exception as e:
    print(f"\n[FAILURE] Error during forward pass with patched model: {e}")
    traceback.print_exc()

print("\nTest finished.")


