import torch
import transformers
# Import the correct model class for Llama-3.2-Vision
from transformers import AutoTokenizer, AutoConfig, MllamaForConditionalGeneration
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
# Using the specified Llama-3.2-Vision model identifier
model_name = "meta-llama/Llama-3.2-11B-Vision" # <<< TARGET MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Using model: {model_name}")

# --- Load Tokenizer ---
print(f"Loading tokenizer for {model_name}...")
try:
    # trust_remote_code might be needed depending on tokenizer implementation
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Ensure tokenizer has a padding token if it doesn't (use EOS)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set tokenizer pad_token to eos_token ({tokenizer.pad_token})")
        else:
            # Add a pad token if both pad and eos are missing
            tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Or another suitable token
            print(f"Added pad_token: {tokenizer.pad_token}")
    print("Tokenizer loaded.")
except Exception as e: print(f"Error loading tokenizer: {e}"); exit(1)

# --- Load Full Model ---
print(f"Loading full model {model_name} (this may take time and VRAM)...")
try:
    # Use MllamaForConditionalGeneration for this model
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True, # Often required for newer/complex models
        low_cpu_mem_usage=True, # Recommended for large models
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    ).to(device)
    model.eval() # Set to evaluation mode
    print("Full model loaded.")
except Exception as e: print(f"Error loading full model: {e}"); traceback.print_exc(); exit(1)

# --- Apply HeadInfer Patch ---
# mp_headinfer should handle MllamaForConditionalGeneration based on the code in headinfer_mllama_support
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
    # Use a generic sentence as the model details might differ slightly
    sentence = "The quick brown fox jumps over the lazy dog. "
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
    tokenizer.padding_side = "left" # Set padding side
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device) # Get mask as well

    # Optional: Limit sequence length if needed
    # Note: Max sequence length for Llama-3.2-Vision might be different, adjust if necessary
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

# --- Run Forward Pass (Patched + OffloadedCache + Long Text) ---
print("\n--- Running Forward Pass (Patched Mllama + OffloadedCache + Long Text) ---")
try:
    with torch.inference_mode():
        start_time = time.time()
        print("Executing model forward pass (patched)...")
        # Call the main model's forward pass with the long input
        # Pass pixel_values=None as we are only testing text path patching
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=None, # Crucial for text-only pass with Mllama
            past_key_values=past_key_values, # Pass the cache object
            use_cache=True # Enable caching
            )
        torch.cuda.synchronize(device) if device == "cuda" else None # Sync for timing
        end_time = time.time()
        # MllamaForConditionalGeneration output structure might differ slightly, access logits correctly
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        print(f"Forward pass successful. Logits shape: {logits.shape}")
        print(f"Execution time: {end_time - start_time:.3f} seconds")
        # Check if cache was populated
        if len(past_key_values.key_cache) > 0:
            print(f"OffloadedCache was populated with {len(past_key_values.key_cache)} layers.")
        else:
            print("Warning: OffloadedCache was not populated.")
        del outputs # Clean up memory

    print("\n[SUCCESS] HeadInfer patch applied and forward pass with long text executed without errors.")

except Exception as e:
    print(f"\n[FAILURE] Error during forward pass with patched model: {e}")
    traceback.print_exc()

print("\nTest finished.")


