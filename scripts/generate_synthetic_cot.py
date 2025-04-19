import json
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
import torch

# Assuming necessary libraries are installed:
# Import the specific model class along with AutoProcessor
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from PIL import Image # Make sure PIL is imported
import bitsandbytes # Needed for 4-bit quantization
import accelerate # Potentially needed for device_map='auto'
import tiktoken # Potentially needed for specific tokenizers like Qwen

# Suppress specific warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_timestamp(ts_str):
    """Parses TVQA timestamp string 'start-end' into float seconds."""
    try:
        start, end = map(float, ts_str.split('-'))
        # Basic validation for timestamp range
        if start >= end or start < 0:
            print(f"Warning: Invalid timestamp range '{ts_str}'. Start: {start}, End: {end}")
            return None, None
        return start, end
    except Exception as e:
        # Catch potential errors during splitting or conversion
        print(f"Warning: Could not parse timestamp string '{ts_str}': {e}")
        return None, None


def load_multimodal_model_and_processor(model_name_or_path, use_4bit=True, device_map="auto"):
    """
    Loads a multimodal model (specifically using Qwen2_5_VLForConditionalGeneration)
    and its processor using AutoProcessor.
    Optionally loads the model in 4-bit. Assumes libraries are installed.
    Sets model.config.use_image_token = True.
    """
    print(f"Loading multimodal model and processor: {model_name_or_path}...")
    print(f"Attempting to use device_map='{device_map}'")

    # Configure 4-bit quantization if requested
    quantization_config = None
    # Determine the optimal dtype based on CUDA availability and capability
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype # Use detected float16/bfloat16
        )
        print(f"Configured for 4-bit quantization with compute dtype: {torch_dtype}.")
    else:
        print("4-bit quantization not requested. Loading in default precision.")


    # Load the processor using AutoProcessor
    try:
        # trust_remote_code=True is often necessary for custom model code
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        print("Processor loaded.")
    except Exception as e:
        print(f"Error loading processor for {model_name_or_path}: {e}")
        print("Hint: Check model name, network connection, and if `trust_remote_code=True` is needed.")
        if "token" in str(e).lower():
             print("Hint: You might need to log in using `huggingface-cli login`.")
        raise e # Re-raise error after printing hints

    # Load the model using the SPECIFIC Qwen2_5_VLForConditionalGeneration class
    model_load_kwargs = {
        "low_cpu_mem_usage": True, # Useful for large models
        "trust_remote_code": True, # Often required for multimodal models
        "device_map": device_map   # Handles device placement (CPU/GPU/multi-GPU)
    }
    if quantization_config:
        model_load_kwargs["quantization_config"] = quantization_config
        # Don't specify torch_dtype when using bitsandbytes 4-bit
        print(f"Attempting to load model in 4-bit using Qwen2_5_VLForConditionalGeneration...")
    else:
        model_load_kwargs["torch_dtype"] = torch_dtype
        print(f"Attempting to load model with dtype: {torch_dtype} using Qwen2_5_VLForConditionalGeneration...")


    try:
        # *** Use the specific class here instead of AutoModelForCausalLM ***
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **model_load_kwargs
        )
        print(f"Loaded model using Qwen2_5_VLForConditionalGeneration class.")

        # *** Set use_image_token based on user suggestion ***
        if hasattr(model, 'config'):
            print("Setting model.config.use_image_token = True")
            model.config.use_image_token = True
        else:
            print("Warning: Model object does not have a 'config' attribute. Cannot set use_image_token.")

        model.eval() # Set model to evaluation mode (disables dropout, etc.)
        print("Model loaded and set to eval mode.")

    except Exception as e:
        # Provide specific hints based on common errors
        print(f"Error loading model {model_name_or_path} using Qwen2_5_VLForConditionalGeneration: {e}")
        # ... (rest of the error handling remains the same) ...
        if "trust_remote_code=True" in str(e):
             print("Hint: The model requires executing code from the repo. Ensure you trust the source.")
        if quantization_config and ("CUDA" in str(e).upper() or "bitsandbytes" in str(e)):
             print("Hint: bitsandbytes 4-bit loading failed. Check CUDA setup, driver compatibility, and bitsandbytes installation.")
        if "out of memory" in str(e).lower():
             print("Hint: Model might be too large for available RAM/VRAM. Try enabling --use_4bit if not already, or use a smaller model.")
        if "requires you to execute the configuration file" in str(e):
             print("Hint: `trust_remote_code=True` was likely needed and used, but another error occurred.")
        if "tiktoken" in str(e).lower():
             print("Hint: This model requires the `tiktoken` library. Ensure it's installed (`pip install tiktoken`).")
        if "Qwen2_5_VLForConditionalGeneration" in str(e) and "not found" in str(e).lower():
             print("Hint: The class 'Qwen2_5_VLForConditionalGeneration' was not found. Ensure your `transformers` library is up-to-date or that the model specified correctly uses `trust_remote_code=True` to load its implementation.")
        raise e # Re-raise error after printing hints

    # Pad token handling: Essential for consistent generation behavior
    if not hasattr(processor, 'tokenizer'):
            print("Warning: Loaded processor does not have a 'tokenizer' attribute. Pad token handling might be incorrect.")
    elif processor.tokenizer.pad_token_id is None:
        if processor.tokenizer.eos_token_id is not None:
            print("Setting tokenizer pad_token_id to eos_token_id.")
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
            # Also update model config if possible and necessary for consistency
            if hasattr(model, 'config') and model.config.pad_token_id is None:
                 model.config.pad_token_id = processor.tokenizer.eos_token_id
        else:
            # This situation can cause issues with batching or generation
            print("Warning: Tokenizer has neither pad_token_id nor eos_token_id set. Generation might fail or behave unexpectedly.")
    else:
        print(f"Tokenizer pad_token_id already set: {processor.tokenizer.pad_token_id}")


    print("Multimodal model and processor loading complete.")
    return model, processor


def generate_cot(model, processor, messages, max_new_tokens=512, temperature=0.2, do_sample=False):
    """
    Generates the CoT using the loaded multimodal model and processor.
    Specifically optimized for Qwen 2.5 VL models.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        print("Preparing structured messages for CoT generation...")
        
        # Extract images from the messages structure
        images_list = []
        for item in messages[0]["content"]:
            if item.get("type") == "image" and "image" in item:
                if isinstance(item["image"], str):
                    # Load the image using PIL if it's a path string
                    try:
                        img = Image.open(item["image"]).convert("RGB")
                        images_list.append(img)
                    except Exception as img_e:
                        print(f"Warning: Failed to load image {item['image']}: {img_e}. Skipping.")
                else:
                    # Image is already a PIL Image object
                    images_list.append(item["image"])
        
        # Apply chat template to format the input for the model
        print(messages)
        # Use the whole messages structure with the chat template
        formatted_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process the formatted text and images together
        inputs = processor(
            text=formatted_text,
            images=images_list,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to the appropriate device
        inputs = inputs.to(model.device)
        print(f"Inputs prepared and moved to device: {model.device}")
        
        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
        }
        
        # Add padding token ID if available
        if processor.tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = processor.tokenizer.pad_token_id
        elif processor.tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = processor.tokenizer.eos_token_id
            
        # Add EOS token ID if available
        if processor.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = processor.tokenizer.eos_token_id
        
        # Generate with inference mode for efficiency
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode only the newly generated tokens
        input_token_len = inputs['input_ids'].shape[1]
        generated_ids = outputs[0, input_token_len:]
        result = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print("CoT generation finished.")
        return result.strip()

    except Exception as e:
        print(f"Error during CoT generation: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating CoT: {e}"


def main(args):
    """Main processing function to orchestrate CoT generation for TVQA."""
    print("Starting CoT generation process...")
    print(f"Arguments: {args}")

    # --- Setup Paths ---
    processed_dir = Path(args.processed_data_dir)
    tvqa_train_path = Path(args.tvqa_train_file)
    output_path = Path(args.output_file)

    # --- Validate Input Paths ---
    if not processed_dir.is_dir():
        print(f"Error: Processed data directory not found: {processed_dir}")
        exit(1)
    if not tvqa_train_path.is_file():
        print(f"Error: TVQA train file not found: {tvqa_train_path}")
        exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_path}")


    # --- 1. Find available processed clips ---
    # Creates a mapping from clip name (vid_name) to the subfolder containing frames
    available_clips = {} # { clip_name: method_folder_name }
    print(f"Scanning for processed clips in: {processed_dir}")
    for clip_dir in processed_dir.iterdir():
        if not clip_dir.is_dir(): continue # Skip non-directory items
        # Define the expected metadata file name
        metadata_file = clip_dir / "metadata_tvqa_text_centric.json"
        if metadata_file.exists():
            # Find subdirectories within the clip directory (expected to contain frames)
            method_dirs = [d for d in clip_dir.iterdir() if d.is_dir()]
            if method_dirs:
                # Assume the first subdirectory found contains the relevant frames/chunks
                available_clips[clip_dir.name] = method_dirs[0].name
            else:
                # Metadata exists, but no frame folder found - issue warning
                print(f"Warning: Metadata found for {clip_dir.name}, but no subfolder containing frames/chunks detected.")


    print(f"Found {len(available_clips)} processed clips with metadata and frame folders.")
    if not available_clips:
        print("Error: No processed clips found matching the expected structure. Exiting.")
        print(f"Expected structure: {processed_dir}/<clip_name>/metadata_tvqa_text_centric.json and {processed_dir}/<clip_name>/<method_folder>/frame_*.jpg")
        return # Exit if no clips to process

    # --- 2. Load Model & Processor (if not skipping generation) ---
    model, processor = None, None
    if not args.skip_generation:
        try:
            # Load the specified model and processor
            model, processor = load_multimodal_model_and_processor(
                args.model_name_or_path,
                use_4bit=args.use_4bit,
                device_map="auto" # Recommended for handling device placement automatically
            )
        except Exception as e:
            # If model loading fails, print error and exit gracefully
            print(f"Fatal: Failed to load model/processor: {e}. Exiting.")
            return
    else:
        print("Skipping model loading and CoT generation as per --skip_generation flag.")

    # --- 3. Process TVQA train file ---
    # Initialize counters for tracking progress and issues
    processed_count = 0
    error_count = 0
    skipped_clips = 0
    skipped_missing_fields = 0
    skipped_bad_ts = 0
    skipped_meta_error = 0
    skipped_answer_idx = 0

    print(f"Starting processing of TVQA file: {tvqa_train_path}")
    # Open input TVQA file and output file
    with open(tvqa_train_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        # Iterate through each line (JSON object) in the TVQA file
        for line_num, line in enumerate(tqdm(infile, desc="Processing TVQA entries")):
            try:
                # Load JSON data from the current line
                data = json.loads(line)
            except json.JSONDecodeError:
                # Handle lines that are not valid JSON
                print(f"Skipping invalid JSON line #{line_num+1}: {line.strip()}")
                error_count += 1
                continue

            # --- Extract data for the current TVQA entry ---
            vid_name = data.get("vid_name")
            qid = data.get("qid", f"unknown_qid_line_{line_num+1}") # Use placeholder if qid is missing

            # Skip if the video clip wasn't found in the processed directory
            if not vid_name or vid_name not in available_clips:
                skipped_clips += 1
                continue

            method_name = available_clips[vid_name] # Get the frame folder name for this clip
            q = data.get("q") # Question text
            # Answer options
            a0, a1, a2, a3, a4 = data.get("a0"), data.get("a1"), data.get("a2"), data.get("a3"), data.get("a4")
            answer_idx = data.get("answer_idx") # Index of the correct answer
            ts = data.get("ts") # Timestamp string 'start-end'

            # Validate that all essential fields are present
            if not all([q, answer_idx is not None, ts, vid_name, qid, a0, a1, a2, a3, a4]):
                print(f"Skipping entry QID {qid}: Missing one or more essential fields (q, a0-4, answer_idx, ts, vid_name).")
                skipped_missing_fields += 1
                error_count += 1
                continue

            # Get the text of the correct answer
            answers = [a0, a1, a2, a3, a4]
            try:
                correct_answer_text = answers[int(answer_idx)]
            except (IndexError, ValueError):
                # Handle cases where answer_idx is invalid
                print(f"Skipping entry QID {qid}: Invalid answer_idx {answer_idx}")
                skipped_answer_idx += 1
                error_count += 1
                continue

            # --- Load Metadata for the specific clip ---
            metadata_file = processed_dir / vid_name / "metadata_tvqa_text_centric.json"
            try:
                with open(metadata_file, 'r', encoding='utf-8') as meta_f:
                    clip_metadata_segments = json.load(meta_f) # Load the list of segments
            except FileNotFoundError:
                 # This should ideally not happen if available_clips logic is correct
                 print(f"Error QID {qid}: Metadata file not found at {metadata_file}. Inconsistency detected.")
                 error_count += 1
                 continue
            except Exception as e:
                # Handle errors during metadata file reading or JSON parsing
                print(f"Skipping entry QID {qid}: Could not load or parse metadata {metadata_file}: {e}")
                skipped_meta_error += 1
                error_count += 1
                continue

            # --- Parse Timestamp ---
            start_sec, end_sec = parse_timestamp(ts)
            if start_sec is None or end_sec is None:
                # Skip if timestamp string is invalid
                skipped_bad_ts += 1
                error_count += 1
                continue

            # --- Build Structured Input (for generate_cot function) ---
            # This list will contain dictionaries for text and image *paths*
            # The generate_cot function will load the images and create the final input
            content_list = []
            # Keep track of the relative paths of images used for this entry
            all_relevant_filenames_relative = []

            found_overlap = False # Flag to track if any relevant segment was found
            # Ensure metadata is a list before iterating
            if not isinstance(clip_metadata_segments, list):
                 print(f"Warning QID {qid}: Metadata content in {metadata_file} is not a list as expected. Skipping segment processing.")
                 clip_metadata_segments = [] # Treat as empty list to avoid crashing

            # Iterate through segments defined in the metadata
            for segment in clip_metadata_segments:
                # Basic validation of segment structure
                if not isinstance(segment, dict):
                    print(f"Warning QID {qid}: Found non-dict item in metadata segment list. Skipping item.")
                    continue

                # Extract segment details
                seg_start = segment.get("text_start_time_sec")
                seg_end = segment.get("text_end_time_sec")
                seg_text = segment.get("text", "").strip() # Get subtitle text, default to empty string
                corresponding_chunks = segment.get("corresponding_chunks", []) # List of frame info dicts

                # Check if the segment's time range overlaps with the query's time range
                if seg_start is not None and seg_end is not None and \
                   max(start_sec, seg_start) < min(end_sec, seg_end): # Standard overlap condition

                    found_overlap = True
                    # Add text part first (subtitle or placeholder)
                    display_text = seg_text if seg_text else "[Visual context without subtitle]"
                    # Include timing in the text for better context understanding by the model
                    content_list.append({"type": "text", "text": f"Segment ({seg_start:.2f}s - {seg_end:.2f}s): {display_text}"})

                    # Prepare and sort corresponding frame/chunk paths for this segment
                    chunk_info_list = [chunk for chunk in corresponding_chunks if isinstance(chunk, dict) and chunk.get("saved_chunk_filename")]

                    if chunk_info_list:
                        # Sort chunks chronologically, preferably by filename (frameXXXX) or time
                        try:
                            # Attempt sorting based on numeric part of filename (e.g., 'frame0012' -> 12)
                            sorted_chunk_info = sorted(chunk_info_list, key=lambda x: int(Path(x["saved_chunk_filename"]).stem.replace('frame','')))
                        except ValueError:
                            # Fallback to sorting by start time if filenames are not standard
                            print(f"Warning QID {qid}: Non-standard frame filenames in segment {seg_start:.2f}-{seg_end:.2f}. Sorting by time.")
                            sorted_chunk_info = sorted(chunk_info_list, key=lambda x: x.get("chunk_start_time_sec", 0))


                        # Add image *paths* to the content list for this segment
                        # The generate_cot function will now load these paths into PIL images
                        images_added_this_segment = 0
                        for chunk_info in sorted_chunk_info:
                            filename_basename = chunk_info["saved_chunk_filename"]
                            # Construct full path needed for loading by the processor
                            full_image_path = processed_dir / vid_name / method_name / filename_basename
                            # Store relative path for output JSON (more portable)
                            relative_image_path = f"{method_name}/{filename_basename}"

                            # Check if the image file actually exists before adding its path
                            if full_image_path.exists() and full_image_path.is_file():
                                # Add the image *path* string; generate_cot will load it
                                content_list.append({"type": "image", "image": str(full_image_path)})
                                all_relevant_filenames_relative.append(relative_image_path)
                                images_added_this_segment += 1
                            else:
                                # Warn if an expected image file is missing
                                print(f"Warning QID {qid}: Image file not found or is not a file: {full_image_path}, skipping path.")

            # --- Handle cases with no overlapping segments or no images ---
            if not found_overlap:
                 print(f"Warning QID {qid}: No overlapping segments found in metadata for time range {start_sec:.2f}s - {end_sec:.2f}s.")
                 # Provide context about the time range even if no specific segments match
                 content_list.append({"type": "text", "text": f"Focused time range: {start_sec:.2f}s - {end_sec:.2f}s. No specific subtitles or frames identified for this exact interval."})

            # Check if any images were successfully added (by checking for 'image' type in content_list)
            if not any(item.get("type") == "image" for item in content_list):
                print(f"Warning QID {qid}: No valid image paths were added to the context for this entry.")
                # Add a note indicating lack of visual information
                content_list.append({"type": "text", "text": "[No visual frames available for this query time.]"})


            # --- Add the final instruction asking for CoT reasoning ---
            # Craft a clear prompt asking for step-by-step reasoning based *only* on provided context
            final_instruction = f"""\n\nBased *only* on the sequence of text segments and visual frames provided above:
Question: "{q}"
Correct Answer: "{correct_answer_text}"

Provide a step-by-step chain of thought reasoning process that logically derives the correct answer from the given context. Start with "Step 1:" and ensure each step explicitly refers to the information presented in the text segments or visual frames."""
            content_list.append({"type": "text", "text": final_instruction})

            # --- Prepare messages structure (still needed to pass to generate_cot) ---
            messages = [{"role": "user", "content": content_list}]

            # --- Generate CoT (if model loaded) ---
            generated_cot = "Generation skipped." # Default value
            if not args.skip_generation and model and processor:
                # Call the generation function which now handles the input formatting via apply_chat_template
                generated_cot = generate_cot(
                    model, processor,
                    messages, # Pass the structured messages list
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.do_sample
                )
                # Check if generation returned an error message
                if generated_cot.startswith("Error generating CoT:"):
                    error_count += 1 # Increment error count if generation failed


            # --- Prepare output data for JSONL ---
            output_data = {
                "qid": qid,
                "vid_name": vid_name,
                "timestamp": ts, # Original timestamp string
                "query_start_sec": start_sec, # Parsed start time
                "query_end_sec": end_sec,     # Parsed end time
                "question": q,
                "answer_options": answers,
                "correct_answer_idx": answer_idx,
                "correct_answer_text": correct_answer_text,
                # Extract text parts from context for readability in output file
                "context_summary": [item['text'] for item in content_list[:-1] if item.get('type') == 'text'], # Exclude the final instruction
                "final_instruction_prompt": final_instruction, # Store the exact instruction used
                "relevant_chunk_filenames": all_relevant_filenames_relative, # List of relative image paths used
                "model_used": args.model_name_or_path if not args.skip_generation else "N/A",
                "generated_cot": generated_cot # The generated reasoning or error message
            }

            # Write the processed data (including CoT) as a JSON line to the output file
            outfile.write(json.dumps(output_data) + '\n')
            processed_count += 1 # Increment count of successfully processed entries

            # --- Check processing limit ---
            if args.limit and processed_count >= args.limit:
                print(f"\nReached processing limit of {args.limit} entries.")
                break # Stop processing more entries

    # --- Print Summary ---
    print("\n--- Processing Summary ---")
    print(f"Successfully processed and wrote {processed_count} entries.")
    print(f"Skipped {skipped_clips} entries due to missing/unmatched video clips.")
    print(f"Skipped {skipped_missing_fields} entries due to missing essential fields in TVQA data.")
    print(f"Skipped {skipped_answer_idx} entries due to invalid answer index.")
    print(f"Skipped {skipped_bad_ts} entries due to invalid/unparsable timestamps.")
    print(f"Skipped {skipped_meta_error} entries due to metadata loading/parsing errors.")
    print(f"Encountered {error_count} total errors/skips during processing (check logs for details).")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate Chain-of-Thought annotations for TVQA using preprocessed data and a multimodal LLM (e.g., Qwen-VL, LLaVA). Assumes necessary libraries are installed.")
    # Required arguments
    parser.add_argument("processed_data_dir", type=str, help="Path to the base directory containing processed clip folders (e.g., 'processed_videos'). Expects structure: <dir>/<clip_name>/metadata_tvqa_text_centric.json and <dir>/<clip_name>/<method_folder>/frame_*.jpg")
    parser.add_argument("tvqa_train_file", type=str, help="Path to the TVQA training JSONL file (e.g., 'tvqa_train.jsonl').")
    parser.add_argument("output_file", type=str, help="Path to save the output JSONL file with generated CoT.")
    # Model and generation arguments
    # Updated default model to a specific Qwen 2.5 VL model
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Hugging Face model name or path (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen-VL-Chat', 'llava-hf/llava-v1.6-mistral-7b-hf').")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate for CoT.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for generation (e.g., 0.2 for more deterministic, 0.7 for more diverse). Use with --do_sample.")
    parser.add_argument("--do_sample", action='store_true', help="Enable sampling. If disabled (default), uses greedy decoding.")
    # Processing control arguments
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N matching TVQA entries.")
    parser.add_argument("--skip_generation", action='store_true', help="Skip loading model and generating CoT (useful for debugging data processing).")
    parser.add_argument("--use_4bit", action='store_true', help="Load the model using 4-bit quantization (requires bitsandbytes and CUDA).")

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Run the main processing function
    main(args)



