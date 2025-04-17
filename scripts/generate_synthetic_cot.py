import json
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
import torch # For device placement
from PIL import Image # For loading images

# Attempt to import transformers, handle error if not installed
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Specific processor might be needed, often loaded via AutoTokenizer or separate class
    # For Qwen-VL, tokenizer often handles text/image processing together
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: `transformers` library not found. Please install it (`pip install transformers torch Pillow`).")
    print("CoT generation functionality will be disabled.")
    TRANSFORMERS_AVAILABLE = False

# Suppress specific warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress FutureWarning from transformers if needed
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_timestamp(ts_str):
    """Parses TVQA timestamp string 'start-end' into float seconds."""
    try:
        start, end = map(float, ts_str.split('-'))
        return start, end
    except Exception as e:
        print(f"Warning: Could not parse timestamp string '{ts_str}': {e}")
        return None, None

# --- MODIFIED Function ---
def load_multimodal_model_and_tokenizer(model_name_or_path, device_map="auto"):
    """Loads Multimodal LLM and Tokenizer using Hugging Face Transformers."""
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Transformers library is not available.")
    print(f"Loading multimodal model: {model_name_or_path}...")

    # Load tokenizer (Qwen tokenizer often handles multimodal aspects)
    # trust_remote_code=True is often necessary for models with custom code
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    # Load model
    # For large models, device_map='auto' is crucial for multi-GPU or CPU offloading
    # Specify torch_dtype for memory efficiency (bf16 if available, else fp16)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map, # Handles device placement
        trust_remote_code=True
    )
    model.eval() # Set model to evaluation mode

    # Qwen tokenizer often doesn't need a separate image processor loaded explicitly
    # The tokenizer's apply_chat_template or direct call handles image processing info
    # We will pass PIL images directly during input preparation.

    # Set padding token if missing (check model specifics, Qwen might handle differently)
    if tokenizer.pad_token_id is None:
         # Common practice, but verify if Qwen uses a different pad/eos strategy
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id


    print("Multimodal model and tokenizer loaded.")
    # Return model and tokenizer (processor might be implicitly handled by tokenizer)
    return model, tokenizer

# --- NEW Function ---
def prepare_multimodal_input_qwen(tokenizer, text_segments_and_images, final_instruction):
    """
    Prepares the multimodal input for Qwen-VL models using their typical format.

    Args:
        tokenizer: The loaded Qwen tokenizer.
        text_segments_and_images (list): A list of tuples, where each tuple is
                                         ('text', "text content") or ('image', "path/to/image.jpg").
                                         Order matters.
        final_instruction (str): The final text instruction (e.g., asking for CoT).

    Returns:
        dict: Inputs ready for model.generate() (e.g., {'input_ids': ..., 'attention_mask': ...})
              Returns None on error.
    """
    query = []
    image_paths_in_query = []

    # Build the query structure expected by Qwen-VL tokenizer
    for item_type, item_content in text_segments_and_images:
        if item_type == 'text':
            query.append({'text': item_content})
        elif item_type == 'image':
            # Check if image path exists before adding
            if Path(item_content).exists():
                 query.append({'image': item_content})
                 image_paths_in_query.append(item_content) # Keep track for debugging/logging
            else:
                 print(f"Warning: Image path not found, skipping: {item_content}")
                 query.append({'text': f'[Image unavailable: {Path(item_content).name}]'}) # Add placeholder text

    # Add the final instruction text
    query.append({'text': final_instruction})

    # Use the tokenizer to prepare the full query (text + images)
    # The Qwen tokenizer handles image loading/processing when paths are provided this way
    try:
        # The `tokenizer` call for Qwen-VL often handles image processing implicitly
        # when given image paths in the structured query.
        # We don't return pixel_values separately usually.
        # Check Qwen documentation for exact arguments if needed.
        # `padding=True`, `truncation=True` might need careful consideration with images.
        # Let's try without padding/truncation initially, assuming prompt fits.
        # Note: This assumes the tokenizer handles image loading via path.
        # If it expects PIL Images, loading needs to happen before this call.
        # Let's refine to load PIL images first for robustness.

        query_with_pil = []
        loaded_image_count = 0
        for item in query:
            if 'image' in item:
                try:
                    img = Image.open(item['image']).convert('RGB')
                    query_with_pil.append({'image': img}) # Pass PIL image
                    loaded_image_count += 1
                except Exception as e:
                    print(f"Error loading image {item['image']}: {e}. Skipping.")
                    # Add placeholder text if image fails to load
                    query_with_pil.append({'text': f'[Error loading image: {Path(item["image"]).name}]'})
            else:
                query_with_pil.append(item) # Keep text items as is

        if loaded_image_count == 0:
             print("Warning: No images were successfully loaded for this prompt.")

        # Use apply_chat_template if available and appropriate for the model version
        # Otherwise, use the direct call which often works for Qwen-VL multimodal
        if hasattr(tokenizer, 'apply_chat_template'):
             # Construct messages in chat format
             messages = [{"role": "user", "content": query_with_pil}]
             inputs = tokenizer.apply_chat_template(
                 messages,
                 add_generation_prompt=True, # Important for generation
                 return_tensors="pt"
             )
        else:
             # Fallback or alternative method if apply_chat_template isn't right
             # This direct call often works for older Qwen-VL or specific setups
             inputs = tokenizer(query_with_pil, return_tensors='pt')


        print(f"DEBUG: Prepared inputs with {loaded_image_count} images.")
        return inputs

    except Exception as e:
        print(f"Error preparing multimodal input for Qwen-VL: {e}")
        # Print query structure for debugging if error occurs
        # print(f"DEBUG: Failed query structure: {query}") # Careful printing potentially large data
        return None


# --- MODIFIED Function ---
def generate_cot(model, tokenizer, # Removed processor, assuming tokenizer handles it
                 multimodal_input_data, # Structured input list
                 final_instruction_text, # Final instruction part
                 device, # Only used for moving inputs if needed
                 max_new_tokens=512):
    """Generates the CoT using the multimodal model."""
    if not TRANSFORMERS_AVAILABLE:
        return "Error: Transformers library not available."

    try:
        # Prepare the structured input using the helper function
        inputs = prepare_multimodal_input_qwen(tokenizer, multimodal_input_data, final_instruction_text)
        if inputs is None:
            return "Error: Failed to prepare multimodal inputs."

        # Move inputs to the correct device (model should already be on device via device_map)
        inputs = inputs.to(model.device)

        # Generate
        # Make sure generation config uses correct EOS token ID(s)
        # Qwen models might have multiple EOS tokens
        eos_token_id = [tokenizer.eos_token_id]
        # Check for specific terminators used by Qwen-VL if needed
        # e.g., tokenizer.convert_tokens_to_ids('<|endoftext|>'), tokenizer.convert_tokens_to_ids('<|im_end|>')
        # Add them to eos_token_id list if necessary based on model card/examples

        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'), # Include attention mask if present
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            eos_token_id=eos_token_id, # Use potentially multiple EOS tokens
            pad_token_id=tokenizer.pad_token_id # Ensure pad token is set
        )

        # Decode response
        # Response includes the prompt, need to decode the generated part only
        # Handle potential differences in output structure based on input type
        input_token_len = inputs['input_ids'].shape[1]
        generated_ids = outputs[0, input_token_len:]
        result = tokenizer.decode(generated_ids, skip_special_tokens=True) # Skip special tokens for cleaner output

        return result.strip()

    except Exception as e:
        print(f"Error during CoT generation: {e}")
        if "CUDA out of memory" in str(e):
             print("CUDA out of memory. Try reducing max_new_tokens or using quantization (e.g., 4-bit).")
        return f"Error generating CoT: {e}"


def main(args):
    """Main processing function."""
    if not TRANSFORMERS_AVAILABLE and not args.skip_generation:
        print("Transformers library needed for generation. Exiting.")
        return

    processed_dir = Path(args.processed_data_dir)
    tvqa_train_path = Path(args.tvqa_train_file)
    output_path = Path(args.output_file)

    # --- 1. Find available processed clips ---
    available_clips = {} # Store clip_name -> method_name mapping
    if processed_dir.is_dir():
        for clip_dir in processed_dir.iterdir():
            if not clip_dir.is_dir(): continue
            metadata_file = clip_dir / "metadata_tvqa_text_centric.json"
            if metadata_file.exists():
                # Find the method directory to locate frames
                method_dirs = [d for d in clip_dir.iterdir() if d.is_dir()]
                if method_dirs:
                    # Use the first method found for locating frames
                    # Assumes only one method was run per clip dir, or uses the first alphabetically
                    available_clips[clip_dir.name] = method_dirs[0].name
                else:
                    print(f"Warning: Metadata found for {clip_dir.name}, but no method subfolder containing frames.")

    print(f"Found {len(available_clips)} processed clips with metadata and frame folders in {processed_dir}")
    if not available_clips:
        print("No processed clips found. Exiting.")
        return

    # --- 2. Load Model (optional) ---
    model, tokenizer = None, None
    if not args.skip_generation:
        # Use device_map='auto' for potentially large models like Qwen-VL
        try:
            model, tokenizer = load_multimodal_model_and_tokenizer(args.model_name_or_path, device_map="auto")
        except Exception as e:
            print(f"Failed to load model: {e}. Exiting.")
            if "authentication" in str(e).lower():
                 print("Hint: You might need to log in using `huggingface-cli login` and accept model terms.")
            elif "out of memory" in str(e).lower():
                 print("Hint: Model might be too large for your GPU RAM. Consider a smaller model or quantization (e.g., using bitsandbytes).")
            return
    else:
        print("Skipping model loading and CoT generation.")

    # --- 3. Process TVQA train file ---
    processed_count = 0
    with open(tvqa_train_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(tqdm(infile, desc="Processing TVQA entries")):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line #{line_num+1}: {line.strip()}")
                continue

            vid_name = data.get("vid_name")
            qid = data.get("qid")

            # Check if we have processed data for this clip
            if not vid_name or vid_name not in available_clips:
                continue

            # Get the method name used for processing this clip
            method_name = available_clips[vid_name]

            # Extract required fields
            q = data.get("q")
            a0, a1, a2, a3, a4 = data.get("a0"), data.get("a1"), data.get("a2"), data.get("a3"), data.get("a4")
            answer_idx = data.get("answer_idx")
            ts = data.get("ts")

            if q is None or answer_idx is None or ts is None or vid_name is None or qid is None \
               or a0 is None or a1 is None or a2 is None or a3 is None or a4 is None:
                print(f"Skipping entry QID {qid or 'Unknown'} due to missing fields.")
                continue

            # Get correct answer
            answers = [a0, a1, a2, a3, a4]
            try:
                correct_answer_text = answers[answer_idx]
            except IndexError:
                print(f"Skipping entry QID {qid}: Invalid answer_idx {answer_idx}")
                continue

            # Load the corresponding metadata
            metadata_file = processed_dir / vid_name / "metadata_tvqa_text_centric.json"
            try:
                with open(metadata_file, 'r', encoding='utf-8') as meta_f:
                    clip_metadata_segments = json.load(meta_f)
            except Exception as e:
                print(f"Skipping entry QID {qid}: Could not load or parse metadata {metadata_file}: {e}")
                continue

            # Parse timestamp
            start_sec, end_sec = parse_timestamp(ts)
            if start_sec is None or end_sec is None:
                print(f"Skipping entry QID {qid}: Invalid timestamp '{ts}'")
                continue

            # --- Extract Multimodal Context ---
            multimodal_context_list = [] # List of ('text', content) or ('image', path)
            all_relevant_filenames_in_order = []

            if not clip_metadata_segments:
                 print(f"Warning QID {qid}: No segments found in metadata.")
            else:
                for segment in clip_metadata_segments:
                    seg_start = segment.get("text_start_time_sec")
                    seg_end = segment.get("text_end_time_sec")
                    seg_text = segment.get("text", "")
                    corresponding_chunks = segment.get("corresponding_chunks", [])

                    # Check if the segment overlaps with the query timestamp
                    if seg_start is not None and seg_end is not None and \
                       max(start_sec, seg_start) < min(end_sec, seg_end):

                        # Add text part
                        segment_label = "Dialogue/Subtitle" if seg_text else "Gap/Silence"
                        display_text = seg_text if seg_text else "[]"
                        multimodal_context_list.append(('text', f"Segment ({seg_start:.2f}s - {seg_end:.2f}s): {display_text}"))

                        # Add corresponding frame *paths*
                        chunk_filenames_relative = [chunk.get("saved_chunk_filename") for chunk in corresponding_chunks if chunk.get("saved_chunk_filename")]

                        if chunk_filenames_relative:
                            # Sort filenames based on frame number
                            try:
                                sorted_chunk_filenames = sorted(chunk_filenames_relative, key=lambda x: int(Path(x).stem.replace('frame','')))
                            except:
                                sorted_chunk_filenames = sorted(chunk_filenames_relative)

                            # Add image paths to the list
                            for fname in sorted_chunk_filenames:
                                # Construct full path: processed_dir / vid_name / method_name / fname
                                # Note: saved_chunk_filename already contains method_name/fname.jpg
                                full_image_path = processed_dir / vid_name / fname
                                multimodal_context_list.append(('image', str(full_image_path)))
                                all_relevant_filenames_in_order.append(fname) # Store relative path for output JSON
                        else:
                            multimodal_context_list.append(('text', "  Frames: [No corresponding processed frames found in metadata for this segment]"))


            # Generate CoT (optional)
            generated_cot = "Generation skipped."
            if not args.skip_generation and model and tokenizer:
                # Define the final instruction part of the prompt
                final_instruction = f"""\n\nBased ONLY on the interleaved context above (text segments and the visual information from the corresponding frames):
Question: "{q}"
Correct Answer: "{correct_answer_text}"

Generate the chain of thought reasoning process step-by-step, starting with "Step 1:" and ensuring the reasoning strictly uses the provided sequential context."""

                # Generate CoT using the multimodal input list and final instruction
                generated_cot = generate_cot(
                    model, tokenizer,
                    multimodal_context_list, # The list of ('text',...) and ('image', path) tuples
                    final_instruction,
                    None, # Device handled by device_map in model loading
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
                # Store a summary of context used, maybe just the text parts?
                "context_summary": [item[1] for item in multimodal_context_list if item[0] == 'text'],
                "relevant_chunk_filenames": all_relevant_filenames_in_order, # List filenames used
                "generated_cot": generated_cot
            }

            # Write to output file
            outfile.write(json.dumps(output_data) + '\n')
            processed_count += 1

            if args.limit and processed_count >= args.limit:
                print(f"Reached processing limit of {args.limit} entries.")
                break

    print(f"\nFinished processing. Generated CoT data for {processed_count} entries.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Chain-of-Thought annotations for TVQA using preprocessed data and a local MULTIMODAL LLM.")
    parser.add_argument("processed_data_dir", type=str, help="Path to the base directory containing processed clip folders (e.g., 'processed_videos').")
    parser.add_argument("tvqa_train_file", type=str, help="Path to the TVQA training JSONL file (e.g., 'tvqa_train.jsonl').")
    parser.add_argument("output_file", type=str, help="Path to save the output JSONL file with generated CoT.")
    # --- Updated Default Model ---
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-VL-Chat", # Using smaller Qwen-VL-Chat as default example
                        help="Hugging Face model name or path for CoT generation (e.g., 'Qwen/Qwen2.5-VL-32B-Instruct', 'Qwen/Qwen-VL-Chat').")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate for CoT.")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N matching TVQA entries.")
    parser.add_argument("--skip_generation", action='store_true', help="Skip loading model and generating CoT (useful for debugging data loading/context extraction).")

    args = parser.parse_args()

    # --- Add dependency check ---
    if not args.skip_generation and not TRANSFORMERS_AVAILABLE:
         print("Error: Transformers library is required for CoT generation. Please install it.")
         exit(1)
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow library is required for image handling. Please install it (`pip install Pillow`).")
        exit(1)


    main(args)
#
# **Key Changes and How it Works:**
#
# 1.  **Model Loading (`load_multimodal_model_and_tokenizer`):**
#     * Updated to use `trust_remote_code=True`, often needed for Qwen models.
#     * Uses `device_map="auto"` for better handling of large models on available hardware (GPU/CPU).
#     * Assumes the Qwen `AutoTokenizer` handles necessary image processing setup internally or during the call.
#
# 2.  **Input Preparation (`prepare_multimodal_input_qwen`):**
#     * This new function takes a list of tuples `[('text', text_content), ('image', image_path), ...]`.
#     * It iterates through this list:
#         * Loads images from paths using PIL.
#         * Creates a structured list `query_with_pil` containing `{'text': ...}` and `{'image': <PIL.Image>}` dictionaries.
#     * It then uses `tokenizer.apply_chat_template` (or a direct call as fallback) to process this list into the format expected by the Qwen model's `generate` method. This step internally handles image preprocessing and embedding image information alongside text tokens.
#
# 3.  **CoT Generation (`generate_cot`):**
#     * The function signature now accepts the structured `multimodal_input_data` list and the `final_instruction_text`.
#     * It calls `prepare_multimodal_input_qwen` to get the model inputs.
#     * It calls `model.generate` with the prepared inputs (which now include image information).
#     * Decoding extracts the generated text (the CoT).
#
# 4.  **Main Logic (`main`):**
#     * Loads the multimodal model and tokenizer.
#     * Iterates through TVQA entries.
#     * Loads your metadata.
#     * Extracts relevant text segments and *image paths* based on the timestamp `ts`.
#     * Builds the `multimodal_context_list` containing the sequence of `('text', ...)` and `('image', path)` tuples.
#     * Defines the `final_instruction` asking for the CoT.
#     * Calls the modified `generate_cot` with the structured list and final instruction.
#     * Saves the results.
#
# **Before Running:**
#
# * **Install Dependencies:** `pip install transformers torch Pillow accelerate bitsandbytes` (bitsandbytes for potential quantization if needed). You might need specific versions depending on the Qwen model. Check its Hugging Face model card.
# * **Model Access:** Ensure you have accepted terms and can download the specific Qwen-VL model you choose (e.g., `Qwen/Qwen2.5-VL-32B-Instruct` or the smaller `Qwen/Qwen-VL-Chat` used as the default example). You might need to log in using `huggingface-cli login`.
# * **VRAM:** Be mindful of VRAM requirements, especially for the 32B model. You might need quantization (like 4-bit loading via `bitsandbytes`) integrated into the `load_multimodal_model_and_tokenizer` function if you run out of memory.
# * **Paths:** Double-check all input/output paths in the command-line arguments.
#
# This script now provides the framework for using a powerful multimodal model like Qwen-VL to generate CoT reasoning directly informed by both the subtitles and the visual content of your processed fram
