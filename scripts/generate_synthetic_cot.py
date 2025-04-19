import json
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
import torch # For device placement
from PIL import Image # For loading images
import subprocess # For dependency check

# Attempt to import transformers, handle error if not installed
try:
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: `transformers` library or specific LLaVA classes not found.")
    print("Please install necessary libraries (`pip install transformers torch Pillow timm bitsandbytes accelerate`).")
    print("CoT generation functionality will be disabled.")
    TRANSFORMERS_AVAILABLE = False
    # Define a dummy class if needed for script structure if skipping generation
    class LlavaOnevisionForConditionalGeneration: pass

# Attempt to import bitsandbytes for 4-bit loading
try:
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    print("Warning: `bitsandbytes` library not found (`pip install bitsandbytes`). 4-bit quantization will be disabled.")
    BITSANDBYTES_AVAILABLE = False


# Suppress specific warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress FutureWarning from transformers if needed
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_timestamp(ts_str):
    """Parses TVQA timestamp string 'start-end' into float seconds."""
    try:
        start, end = map(float, ts_str.split('-'))
        # Add basic validation
        if start >= end or start < 0:
             print(f"Warning: Invalid timestamp range '{ts_str}'. Start: {start}, End: {end}")
             return None, None
        return start, end
    except Exception as e:
        print(f"Warning: Could not parse timestamp string '{ts_str}': {e}")
        return None, None

# --- MODIFIED Function ---
def load_multimodal_model_and_processor(model_name_or_path, use_4bit=True, device_map="auto"):
    """
    Loads LLaVA-OneVision model (optionally in 4-bit) and its Processor.
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Transformers library is not available.")
    print(f"Loading multimodal model and processor: {model_name_or_path}...")

    # Configure 4-bit quantization if requested and available
    quantization_config = None
    if use_4bit:
        if BITSANDBYTES_AVAILABLE:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 # Or bfloat16 if supported and preferred
            )
            print("Configured for 4-bit quantization.")
        else:
            print("Warning: 4-bit quantization requested but bitsandbytes is not available. Loading in default precision.")

    # Load the processor
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        print("Processor loaded.")
    except Exception as e:
        print(f"Error loading processor for {model_name_or_path}: {e}")
        raise e

    # Load the model
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Attempting to load model with dtype: {dtype if not quantization_config else '4-bit'}...")
    try:
        # Try specific class first if needed, fallback to AutoModel
        ModelClass = LlavaOnevisionForConditionalGeneration # Default to specific class
        try:
             # Check if class exists before using it directly
             from transformers import LlavaOnevisionForConditionalGeneration
        except ImportError:
             print("LlavaOnevisionForConditionalGeneration not found, using AutoModelForCausalLM.")
             ModelClass = AutoModelForCausalLM

        model = ModelClass.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype if not quantization_config else None, # dtype ignored if quantizing
            low_cpu_mem_usage=True,
            device_map=device_map,
            quantization_config=quantization_config, # Pass quantization config
            trust_remote_code=True
        )
        print(f"Loaded using {ModelClass.__name__} class.")

        model.eval() # Set model to evaluation mode
        print("Model loaded and set to eval mode.")
    except Exception as e:
        print(f"Error loading model {model_name_or_path}: {e}")
        if "requires you to execute the configuration file" in str(e):
             print("Hint: `trust_remote_code=True` might be needed.")
        if quantization_config and "CUDA setup failed" in str(e):
             print("Hint: bitsandbytes might not be correctly installed or configured for your CUDA version.")
        raise e

    # Pad token handling
    if not hasattr(processor, 'tokenizer'):
         print("Warning: Processor does not have a 'tokenizer' attribute.")
    elif processor.tokenizer.pad_token is None:
        print("Setting pad_token to eos_token in tokenizer.")
        try:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            if hasattr(model, 'config') and model.config.pad_token_id is None:
                 model.config.pad_token_id = processor.tokenizer.eos_token_id
        except Exception as e:
            print(f"Warning: Could not set pad_token: {e}")

    print("Multimodal model and processor loading complete.")
    return model, processor

# --- REMOVED Function ---
# def prepare_llava_prompt_string_and_images(...):
# This function is no longer needed as apply_chat_template handles it.

# --- MODIFIED Function ---
def generate_cot(model, processor, # Takes processor now
                 messages, # Structured conversation history (list of dicts)
                 max_new_tokens=512):
    """
    Generates the CoT using the multimodal model (LLaVA-OneVision style).
    MODIFIED: Uses processor.apply_chat_template directly. Includes torch.inference_mode().
    """
    if not TRANSFORMERS_AVAILABLE:
        return "Error: Transformers library not available."

    try:
        # --- Step 1: Prepare inputs using apply_chat_template ---
        # This function now handles text formatting, image loading/processing, and tokenization
        print(f"DEBUG: Calling apply_chat_template with {len(messages[0]['content'])} content items.")
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True, # Crucial for generation
            tokenize=True,              # Ensure output is tokenized
            return_tensors="pt"         # Return PyTorch tensors
        )
        # print(f"DEBUG: Final processor output keys: {inputs.keys()}") # Debug

        # Move inputs to the correct device
        inputs = inputs.to(model.device)

        # --- Step 2: Generation using standard Transformers generate ---
        print("Attempting CoT generation using standard Transformers generate...")

        eos_token_id = processor.tokenizer.eos_token_id
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        possible_eos_tokens = [eos_token_id]
        # Add other EOS tokens if needed

        # *** Use torch.inference_mode() for efficiency ***
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, # Pass processor outputs directly
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=False, # Use greedy for CoT
                eos_token_id=possible_eos_tokens,
                pad_token_id=pad_token_id
            )

        # Decode only the newly generated part
        input_token_len = inputs['input_ids'].shape[1]
        generated_ids = outputs[0, input_token_len:]
        result = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return result.strip()

    except Exception as e:
        print(f"Error during CoT generation: {e}")
        import traceback
        traceback.print_exc()
        if "CUDA out of memory" in str(e):
             print("CUDA out of memory. Try reducing number of images in context or ensure 4-bit loading is active.")
        return f"Error generating CoT: {e}"


# --- MODIFIED Function ---
def main(args):
    """Main processing function."""
    if not TRANSFORMERS_AVAILABLE and not args.skip_generation:
        print("Transformers library needed for generation. Exiting.")
        return
    # Check for bitsandbytes if 4bit requested
    if args.use_4bit and not BITSANDBYTES_AVAILABLE:
         print("Error: --use_4bit requested, but bitsandbytes library is not available. Please install it.")
         exit(1)


    processed_dir = Path(args.processed_data_dir)
    tvqa_train_path = Path(args.tvqa_train_file)
    output_path = Path(args.output_file)

    # --- 1. Find available processed clips ---
    available_clips = {} # Store clip_name -> method_name mapping
    if processed_dir.is_dir():
        for clip_dir in processed_dir.iterdir():
            if not clip_dir.is_dir(): continue
            metadata_file = clip_dir / "metadata_tvqa_text_centric.json" # Correct filename
            if metadata_file.exists():
                method_dirs = [d for d in clip_dir.iterdir() if d.is_dir()]
                if method_dirs:
                    available_clips[clip_dir.name] = method_dirs[0].name
                else:
                    print(f"Warning: Metadata found for {clip_dir.name}, but no method subfolder containing frames.")

    print(f"Found {len(available_clips)} processed clips with metadata and frame folders in {processed_dir}")
    if not available_clips:
        print("No processed clips found. Exiting.")
        return

    # --- 2. Load Model & Processor (optional) ---
    model, processor = None, None # Using processor now
    if not args.skip_generation:
        try:
            model, processor = load_multimodal_model_and_processor(
                args.model_name_or_path,
                use_4bit=args.use_4bit, # Pass quantization flag
                device_map="auto"
            )
        except Exception as e:
            print(f"Failed to load model/processor: {e}. Exiting.")
            if "authentication" in str(e).lower():
                 print("Hint: You might need to log in using `huggingface-cli login` and accept model terms.")
            elif "out of memory" in str(e).lower():
                 print("Hint: Model might be too large for your GPU RAM. Consider enabling --use_4bit or using a smaller model.")
            return
    else:
        print("Skipping model loading and CoT generation.")

    # --- 3. Process TVQA train file ---
    processed_count = 0
    error_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tvqa_train_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(tqdm(infile, desc="Processing TVQA entries")):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line #{line_num+1}: {line.strip()}")
                error_count += 1
                continue

            vid_name = data.get("vid_name")
            qid = data.get("qid")

            if not vid_name or vid_name not in available_clips:
                continue

            method_name = available_clips[vid_name]
            q = data.get("q")
            a0, a1, a2, a3, a4 = data.get("a0"), data.get("a1"), data.get("a2"), data.get("a3"), data.get("a4")
            answer_idx = data.get("answer_idx")
            ts = data.get("ts")

            if not all([q, answer_idx is not None, ts, vid_name, qid, a0, a1, a2, a3, a4]):
                print(f"Skipping entry QID {qid or 'Unknown'} due to missing fields.")
                error_count += 1
                continue

            answers = [a0, a1, a2, a3, a4]
            try:
                correct_answer_text = answers[answer_idx]
            except IndexError:
                print(f"Skipping entry QID {qid}: Invalid answer_idx {answer_idx}")
                error_count += 1
                continue

            metadata_file = processed_dir / vid_name / "metadata_tvqa_text_centric.json"
            try:
                with open(metadata_file, 'r', encoding='utf-8') as meta_f:
                    clip_metadata_segments = json.load(meta_f)
            except Exception as e:
                print(f"Skipping entry QID {qid}: Could not load or parse metadata {metadata_file}: {e}")
                error_count += 1
                continue

            start_sec, end_sec = parse_timestamp(ts)
            if start_sec is None or end_sec is None:
                print(f"Skipping entry QID {qid}: Invalid timestamp '{ts}'")
                error_count += 1
                continue

            # --- Build Structured Input for apply_chat_template ---
            content_list = [] # Build the list for the 'content' field
            all_relevant_filenames_relative = [] # Store relative paths for output JSON

            if not clip_metadata_segments:
                 print(f"Warning QID {qid}: No segments found in metadata.")
                 # Add a placeholder text if no segments overlap?
                 content_list.append({"type": "text", "text": f"Context for time {start_sec:.2f}s - {end_sec:.2f}s (No specific subtitles/frames found in metadata)"})
            else:
                found_overlap = False
                for segment in clip_metadata_segments:
                    seg_start = segment.get("text_start_time_sec")
                    seg_end = segment.get("text_end_time_sec")
                    seg_text = segment.get("text", "")
                    corresponding_chunks = segment.get("corresponding_chunks", [])

                    # Check if the segment overlaps with the query timestamp
                    if seg_start is not None and seg_end is not None and \
                       max(start_sec, seg_start) < min(end_sec, seg_end):

                        found_overlap = True
                        # Add text part first
                        segment_label = "Dialogue/Subtitle" if seg_text else "Gap/Silence"
                        display_text = seg_text if seg_text else "[]"
                        content_list.append({"type": "text", "text": f"Segment ({seg_start:.2f}s - {seg_end:.2f}s): {display_text}"})

                        # Add corresponding frame *paths* after the text for this segment
                        chunk_info_list = [chunk for chunk in corresponding_chunks if chunk.get("saved_chunk_filename")]

                        if chunk_info_list:
                            # Sort chunks within the segment
                            try:
                                sorted_chunk_info = sorted(chunk_info_list, key=lambda x: int(Path(x["saved_chunk_filename"]).stem.replace('frame','')))
                            except:
                                sorted_chunk_info = sorted(chunk_info_list, key=lambda x: x["chunk_start_time_sec"])

                            # Add image paths to the list
                            image_paths_added_segment = []
                            for chunk_info in sorted_chunk_info:
                                filename_basename = chunk_info["saved_chunk_filename"]
                                full_image_path = processed_dir / vid_name / method_name / filename_basename
                                relative_image_path = f"{method_name}/{filename_basename}"

                                if full_image_path.exists():
                                     # Add image type with the *path string* for apply_chat_template
                                     content_list.append({"type": "image", "image": str(full_image_path)})
                                     all_relevant_filenames_relative.append(relative_image_path)
                                     image_paths_added_segment.append(str(full_image_path))
                                else:
                                     print(f"Warning QID {qid}: Image file not found {full_image_path}, skipping.")
                                     content_list.append({"type": "text", "text": f"[Image file not found: {relative_image_path}]"})
                            # Optional: Add text indicating which frames were just shown
                            # if image_paths_added_segment:
                            #      content_list.append({"type": "text", "text": f"(Frames corresponding to above segment: {len(image_paths_added_segment)} images)"})

                if not found_overlap:
                     content_list.append({"type": "text", "text": f"Context for time {start_sec:.2f}s - {end_sec:.2f}s (No specific subtitles/frames found overlapping in metadata)"})


            # Add the final instruction asking for CoT
            final_instruction = f"""\n\nBased ONLY on the interleaved context above (text segments and the visual information from the corresponding frames):
Question: "{q}"
Correct Answer: "{correct_answer_text}"

Generate the chain of thought reasoning process step-by-step, starting with "Step 1:" and ensuring the reasoning strictly uses the provided sequential context."""
            content_list.append({"type": "text", "text": final_instruction})

            # Final structure for apply_chat_template
            messages = [{"role": "user", "content": content_list}]

            # Generate CoT (optional)
            generated_cot = "Generation skipped."
            if not args.skip_generation and model and processor:
                # Call generate_cot with the messages structure
                generated_cot = generate_cot(
                    model, processor,
                    messages, # Pass the structured messages list
                    # final_instruction no longer needed here, it's in messages
                    None, # Device handled by model's device_map
                    args.max_new_tokens
                )

            # Prepare output data
            output_data = {
                "qid": qid,
                "vid_name": vid_name,
                "timestamp": ts,
                "question": q,
                "answer_options": answers,
                "correct_answer_idx": answer_idx,
                "correct_answer_text": correct_answer_text,
                "context_summary": [item['text'] for item in content_list if item['type'] == 'text'], # Extract text parts
                "relevant_chunk_filenames": all_relevant_filenames_relative,
                "generated_cot": generated_cot
            }

            outfile.write(json.dumps(output_data) + '\n')
            processed_count += 1

            if args.limit and processed_count >= args.limit:
                print(f"Reached processing limit of {args.limit} entries.")
                break

    print(f"\nFinished processing. Generated CoT data for {processed_count} entries.")
    if error_count > 0:
        print(f"Encountered {error_count} errors or skipped entries during processing (check logs).")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Chain-of-Thought annotations for TVQA using preprocessed data and a local MULTIMODAL LLM (LLaVA-style).")
    parser.add_argument("processed_data_dir", type=str, help="Path to the base directory containing processed clip folders (e.g., 'processed_videos').")
    parser.add_argument("tvqa_train_file", type=str, help="Path to the TVQA training JSONL file (e.g., 'tvqa_train.jsonl').")
    parser.add_argument("output_file", type=str, help="Path to save the output JSONL file with generated CoT.")
    parser.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", # Placeholder - Use actual ID
                        help="Hugging Face model name or path for CoT generation (e.g., 'llava-hf/llava-onevision-qwen2-7b-ov-hf', 'llava-hf/llava-v1.6-mistral-7b-hf').")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate for CoT.")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N matching TVQA entries.")
    parser.add_argument("--skip_generation", action='store_true', help="Skip loading model and generating CoT.")
    # --- Added 4-bit flag ---
    parser.add_argument("--use_4bit", action='store_true', help="Load the model using 4-bit quantization (requires bitsandbytes).")


    args = parser.parse_args()

    # --- Add dependency check ---
    if not args.skip_generation and not TRANSFORMERS_AVAILABLE:
         print("Error: Transformers library is required for CoT generation. Please install it.")
         exit(1)
    if args.use_4bit and not BITSANDBYTES_AVAILABLE:
         print("Error: --use_4bit requested, but bitsandbytes library is not found. Please install it (`pip install bitsandbytes`).")
         exit(1)
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow library is required for image handling. Please install it (`pip install Pillow`).")
        exit(1)


    main(args)
# **Summary of Changes:**
#
# 1.  **`load_multimodal_model_and_processor`:**
#     * Added `use_4bit` argument.
#     * Added `bitsandbytes` import check.
#     * Configures `BitsAndBytesConfig` and passes it to `from_pretrained` if `use_4bit` is True and `bitsandbytes` is available.
# 2.  **Removed `prepare_llava_prompt_string_and_images`:** This function is no longer needed as `apply_chat_template` handles the formatting.
# 3.  **`generate_cot`:**
#     * Changed signature to accept the structured `messages` list instead of separate context components.
#     * **Removed** the manual prompt building logic.
#     * **Added** the call to `processor.apply_chat_template(messages, ..., return_tensors="pt")` to get the final `inputs` dictionary.
#     * **Added** `with torch.inference_mode():` around the `model.generate()` call.
# 4.  **`main`:**
#     * **Builds `messages` list:** Iterates through the relevant segments and chunks, creating the list of dictionaries (`{"type": "text", "text": ...}` or `{"type": "image", "image": path_string}`) as required by `apply_chat_template`.
#     * Calls the modified `generate_cot` function, passing the `messages` list.
#     * Added `--use_4bit` command-line argument.
#
# This version now correctly uses the modern `apply_chat_template` method for multimodal input preparation, as shown in the documentation you provided, and incorporates your requests for 4-bit loading and inference mode. Remember to install `bitsandbytes` (`pip install bitsandbytes`) if you want to use the `--use_4bit` fl

