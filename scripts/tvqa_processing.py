import cv2
import os
import json
import argparse
import warnings
import pysrt

import numpy as np
from pathlib import Path
from tqdm import tqdm

from video_preprocessing import (
    # Frame averaging functions
    weighted_average,
    weighted_average_exponential,
    weighted_average_ramp,
    blend_blur_with_last_frame,
    generate_srt_from_audio
)

# ==============================================================================
# TVQA-Specific Helper Functions
# ==============================================================================

def read_tvqa_frame_chunks(frame_dir, chunk_size, resize_to=None, skip_block=0):
    """
    Generator yielding lists of `chunk_size` frames read from image files
    in a directory (sorted numerically), skipping `skip_block` frames between chunks.
    Designed for datasets like TVQA where frames are stored as individual images.

    Args:
        frame_dir (Path): Directory containing frame image files (e.g., frame_00001.jpg).
        chunk_size (int): Number of frames per chunk.
        resize_to (tuple or None): Target (width, height) for resizing.
        skip_block (int): Number of frames to skip between chunks.

    Yields:
        list: A list of numpy arrays representing a chunk of frames, or None if error.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    try:
        # List and sort frame files numerically based on standard frame naming
        # Assumes names like frame_xxxx.jpg or similar that sort correctly alphabetically/numerically
        # Adjust glob pattern if TVQA uses a different naming scheme
        frame_files = sorted(frame_dir.glob("*.jpg"))
        if not frame_files:
             print(f"Warning: No .jpg frames found in {frame_dir}")
             return # Yield nothing if no frames
    except Exception as e:
        print(f"Error listing or sorting frames in {frame_dir}: {e}")
        return # Yield nothing on error

    current_frame_index = 0 # Overall index in the full frame sequence
    chunk = []
    total_cycle_len = chunk_size + skip_block
    if total_cycle_len == 0: total_cycle_len = 1

    while current_frame_index < len(frame_files):
        cycle_pos = current_frame_index % total_cycle_len

        if cycle_pos < chunk_size:
            # Read frame file
            frame_path = frame_files[current_frame_index]
            try:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    print(f"Warning: Failed to read frame {frame_path}. Skipping.")
                    # Decide how to handle: skip frame, stop processing?
                    # Let's skip this frame and continue the cycle for robustness
                    current_frame_index += 1
                    continue

                if resize_to:
                    frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
                # Store as float32 for potential averaging calculations
                chunk.append(frame.astype(np.float32))

                # If chunk is full, yield it
                if len(chunk) == chunk_size:
                    yield chunk
                    chunk = [] # Reset for next chunk

            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                # Decide how to handle: skip frame, stop processing?
                # Let's skip this frame
        # else: # Skipping phase (just increment index, no file read needed)
        #     pass

        current_frame_index += 1

    # Yield any remaining partial chunk at the end
    if chunk:
        yield chunk


def handle_subtitles_tvqa(tvqa_sub_path, # Path to PREFERABLY use TVQA's provided subs
                          audio_path, # Path to audio file for fallback generation
                          output_srt_path, # Where to save generated SRT if needed
                          whisper_model,
                          force_subtitle_generation=False):
    """
    Manages subtitle acquisition specifically for TVQA.
    1. Tries to parse the provided TVQA subtitle file.
    2. If missing or forced, generates subtitles from the audio file using Whisper.

    Args:
        tvqa_sub_path (Path or None): Path to the ground truth subtitle file provided by TVQA.
        audio_path (Path or None): Path to the corresponding audio file (e.g., MP3).
        output_srt_path (Path): Path where generated SRT file should be saved if generation occurs.
        whisper_model (str): Name of the Whisper model to use for generation.
        force_subtitle_generation (bool): If True, always generate, ignore tvqa_sub_path.

    Returns:
        list: A list of parsed subtitle tuples (start_sec, end_sec, text), sorted by start time.
    """
    parsed_subtitles = []
    sub_file_to_parse = None

    # Decide whether to generate or use existing
    should_generate = force_subtitle_generation
    if not force_subtitle_generation:
        if tvqa_sub_path and tvqa_sub_path.exists():
            print(f"Found provided TVQA subtitle file: {tvqa_sub_path}")
            sub_file_to_parse = tvqa_sub_path
        elif audio_path and audio_path.exists():
            print(f"Provided TVQA subtitle file not found or specified. Will attempt generation from audio: {audio_path}")
            should_generate = True
        else:
            print(f"Warning: No provided subtitle file ({tvqa_sub_path}) and no audio file ({audio_path}). Cannot get subtitles.")
            return [] # Cannot proceed

    # Generate if needed
    if should_generate:
        if audio_path and audio_path.exists():
            # Generate SRT from audio, save to output_srt_path
            gen_success = generate_srt_from_audio(str(audio_path), str(output_srt_path), model_name=whisper_model)
            if gen_success:
                sub_file_to_parse = output_srt_path # Parse the file we just created
            else:
                print(f"Warning: Whisper generation failed for audio {audio_path}.")
                # Fallback: if tvqa_sub_path existed, try using that even if generation failed
                if tvqa_sub_path and tvqa_sub_path.exists():
                     print("Falling back to using provided TVQA subtitle file.")
                     sub_file_to_parse = tvqa_sub_path
                else:
                     return [] # Generation failed, no fallback
        else:
             print(f"Warning: Cannot generate subtitles. Audio file not found or not provided: {audio_path}")
             # Fallback logic if needed... but likely indicates an issue upstream
             if tvqa_sub_path and tvqa_sub_path.exists():
                 print("Falling back to using provided TVQA subtitle file.")
                 sub_file_to_parse = tvqa_sub_path
             else:
                 return []


    # Parse the selected subtitle file (either provided or generated)
    if sub_file_to_parse and sub_file_to_parse.exists():
        print(f"Parsing subtitles from: {sub_file_to_parse} using pysrt")
        try:
            subs = pysrt.open(str(sub_file_to_parse), encoding='utf-8')
            subs.sort(key=lambda x: x.start)
            for sub in subs:
                start_sec = sub.start.ordinal / 1000.0
                end_sec = sub.end.ordinal / 1000.0
                clean_text = sub.text.replace('\n', ' ').strip()
                parsed_subtitles.append((start_sec, end_sec, clean_text))
            if parsed_subtitles:
                 print(f"Successfully parsed {len(parsed_subtitles)} subtitle entries.")
            else:
                 print(f"Warning: pysrt parsed 0 subtitle entries from {sub_file_to_parse}.")
        except Exception as e:
            print(f"Error parsing SRT file {sub_file_to_parse} with pysrt: {e}")
            parsed_subtitles = []
    elif sub_file_to_parse and not sub_file_to_parse.exists():
         print(f"Error: Subtitle file selected for parsing does not exist: {sub_file_to_parse}")
    elif not sub_file_to_parse:
         print("Warning: No subtitle file was selected for parsing.")


    return parsed_subtitles


def process_video_chunks_tvqa(frame_dir, # Directory of frames for one clip
                              video_base_out_dir, # Base output dir for this clip
                              num_frames, # Frames per chunk (e.g., 3 for 1 sec @ 3fps)
                              skip_frames, # Frames to skip between chunks
                              resize_to,
                              chunk_weight_function):
    """
    Processes TVQA frame chunks from a directory, applies averaging, saves frames,
    and collects chunk metadata. Uses TVQA's assumed FPS.

    Args:
        frame_dir (Path): Directory containing frame image files for the clip.
        video_base_out_dir (Path): Base output directory for this clip's processed files.
        num_frames (int): Number of frames per chunk.
        skip_frames (int): Number of frames to skip between chunks.
        resize_to (tuple or None): Target dimensions for resizing frames.
        chunk_weight_function (callable): Function (imported) to average frames.

    Returns:
        list: A list of metadata dictionaries for each processed chunk.
    """
    chunk_metadata_list = []
    total_frames_processed_in_video = 0
    frame_method_name = chunk_weight_function.__name__
    # --- Use TVQA's known FPS ---
    fps = 3.0

    print(f"Processing TVQA frames from {frame_dir.name} with chunk_size={num_frames}, skip={skip_frames}, method='{frame_method_name}'...")
    # Use the TVQA-specific frame reader
    for i, chunk in enumerate(read_tvqa_frame_chunks(frame_dir, num_frames, resize_to, skip_frames)):
        if not chunk: continue

        start_cycle_frame_index = i * (num_frames + skip_frames)
        start_frame_num = start_cycle_frame_index + 1
        end_frame_num = start_cycle_frame_index + num_frames

        chunk_start_time = start_cycle_frame_index / fps
        chunk_end_time = (start_cycle_frame_index + num_frames) / fps
        total_frames_processed_in_video += len(chunk)

        # Apply Motion Blur/Averaging (using imported function)
        blur = chunk_weight_function(chunk)
        if blur is None:
            print(f"Warning: Frame averaging failed for chunk {i} (starts frame {start_frame_num})")
            continue

        # Save Motion Blur Frame
        output_image_filename = f"frame{i+1:05d}.jpg" # Consistent 5-digit padding for output chunk index
        relative_image_path = f"{frame_method_name}/{output_image_filename}"
        out_path_img = video_base_out_dir / relative_image_path
        out_path_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path_img), blur)

        # Create frame range string (based on original frame indices)
        # Note: TVQA frame names might be different, adjust if needed
        image_file_range_str = f"source_frame{start_frame_num:05d}-source_frame{end_frame_num:05d}"

        chunk_metadata = {
            "image_file_range": image_file_range_str,
            "saved_image_file": relative_image_path,
            "chunk_start_time_sec": round(chunk_start_time, 3),
            "chunk_end_time_sec": round(chunk_end_time, 3)
        }
        chunk_metadata_list.append(chunk_metadata)

    print(f"Processed {total_frames_processed_in_video} frames into {len(chunk_metadata_list)} chunks.")
    return chunk_metadata_list

def process_single_tvqa_clip(clip_name, # e.g., "s01e01_seg01_clip_00"
                             frame_folder_path, # FULL path to the frame folder for this clip
                             audio_path, # FULL path to the audio file
                             provided_srt_path, # FULL path to the provided SRT file
                             base_output_dir, # Base directory for ALL processed output
                             num_frames_per_chunk,
                             skip_frames,
                             resize_to,
                             chunk_weight_function,
                             whisper_model,
                             force_subtitle_generation=False):
    """
    Orchestrates the processing pipeline for a single TVQA clip using specific paths.
    Saves output under base_output_dir/clip_name/
    """
    print(f"\n--- Processing TVQA Clip: {clip_name} ---")
    # Define output directory for this specific clip
    clip_output_dir = base_output_dir / clip_name
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    # Define where generated SRTs for this clip should be saved
    generated_srt_output_path = clip_output_dir / f"{clip_name}.srt"

    # Validate input paths (frame folder checked by caller)
    if not audio_path.exists():
        print(f"⚠️ Audio file not found: {audio_path}. Subtitle generation might fail.")
    if not provided_srt_path.exists():
         print(f"Note: Provided subtitle file not found: {provided_srt_path}. Will rely on generation if needed.")

    # Step 1: Handle Subtitles (using TVQA-specific logic)
    parsed_subtitles = handle_subtitles_tvqa(
        provided_srt_path, # Pass the specific path
        audio_path,        # Pass the specific path
        generated_srt_output_path, # Where to save if generated
        whisper_model,
        force_subtitle_generation=force_subtitle_generation
    )

    # Step 2: Process Video Frames (using TVQA-specific logic)
    chunk_metadata_list = process_video_chunks_tvqa(
        frame_folder_path, # Pass the specific frame folder path
        clip_output_dir,   # Pass the clip's output dir
        num_frames_per_chunk,
        skip_frames,
        resize_to,
        chunk_weight_function
    )

    # Step 3: Assemble Simplified Text-Centric Metadata Structure including Gaps
    final_metadata_structure = []
    last_event_end_time = 0.0
    epsilon = 1e-3

    if not chunk_metadata_list:
        print("Skipping metadata assembly: No video chunks were processed.")
    else:
        print("Assembling text-centric metadata including gaps...")
        # Helper function to find overlapping chunks
        def find_overlapping_chunks(interval_start, interval_end):
            simplified_chunks = []
            for chunk_info in chunk_metadata_list:
                chunk_start = chunk_info["chunk_start_time_sec"]
                chunk_end = chunk_info["chunk_end_time_sec"]
                if max(interval_start, chunk_start) < min(interval_end, chunk_end):
                    filename_basename = Path(chunk_info["saved_image_file"]).name
                    simplified_chunks.append({
                        "image_file_range": chunk_info["image_file_range"],
                        "saved_chunk_filename": filename_basename,
                        "chunk_start_time_sec": chunk_info["chunk_start_time_sec"],
                        "chunk_end_time_sec": chunk_info["chunk_end_time_sec"]
                    })
            return simplified_chunks

        # Iterate through subtitles to process gaps and subtitle segments
        for sub_start, sub_end, sub_text in parsed_subtitles:
            if sub_start is None or sub_end is None or sub_start >= sub_end:
                print(f"Skipping invalid subtitle segment: start={sub_start}, end={sub_end}, text='{sub_text[:20]}...'")
                continue

            if sub_start > last_event_end_time + epsilon:
                gap_start = last_event_end_time
                gap_end = sub_start
                corresponding_chunks_gap = find_overlapping_chunks(gap_start, gap_end)
                if corresponding_chunks_gap:
                    final_metadata_structure.append({
                        "text": "",
                        "text_start_time_sec": round(gap_start, 3),
                        "text_end_time_sec": round(gap_end, 3),
                        "corresponding_chunks": corresponding_chunks_gap
                    })
                last_event_end_time = gap_end # Correctly update marker

            corresponding_chunks_sub = find_overlapping_chunks(sub_start, sub_end)
            final_metadata_structure.append({
                "text": sub_text,
                "text_start_time_sec": round(sub_start, 3),
                "text_end_time_sec": round(sub_end, 3),
                "corresponding_chunks": corresponding_chunks_sub
            })
            last_event_end_time = max(last_event_end_time, sub_end) # Use max for safety

        last_chunk_end_time = chunk_metadata_list[-1]['chunk_end_time_sec'] if chunk_metadata_list else 0.0
        if last_chunk_end_time > last_event_end_time + epsilon:
            gap_start = last_event_end_time
            gap_end = last_chunk_end_time
            corresponding_chunks_final_gap = find_overlapping_chunks(gap_start, gap_end)
            if corresponding_chunks_final_gap:
                final_metadata_structure.append({
                    "text": "",
                    "text_start_time_sec": round(gap_start, 3),
                    "text_end_time_sec": round(gap_end, 3),
                    "corresponding_chunks": corresponding_chunks_final_gap
                })
        print(f"Assembled metadata for {len(final_metadata_structure)} segments (including gaps).")


    # Step 4: Save Final Metadata Structure to JSON in the clip's output directory
    if final_metadata_structure:
        metadata_path = clip_output_dir / "metadata_tvqa_text_centric.json" # Specific filename
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(final_metadata_structure, f, indent=4, ensure_ascii=False)
            print(f"TVQA text-centric metadata saved to: {metadata_path}")
        except Exception as e:
            print(f"Error saving TVQA metadata to {metadata_path}: {e}")
    else:
        print(f"No text-centric metadata generated for {clip_name}.")

    print(f"--- Finished {clip_name} ---")


# --- MODIFIED Main Loop Function ---
def process_tvqa_dataset(tvqa_base_dir, # Base directory of the downloaded TVQA dataset
                         output_dir, # Base directory for processed output
                         num_frames_per_chunk=3, # e.g., 3 frames = 1 second at 3fps
                         skip_frames=0,
                         resize_to=(224, 224),
                         chunk_weight_function=blend_blur_with_last_frame, # Choose function
                         whisper_model="base",
                         force_subtitle_generation=False,
                         clip_limit=None): # Optional: limit number of clips to process
    """
    Main function to iterate through the TVQA dataset structure (including show folders)
    and process clips. Adjust paths based on your extracted TVQA layout.
    """
    tvqa_base_dir = Path(tvqa_base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Define paths based on user's description ---
    tvqa_hq_frames_dir = tvqa_base_dir / "frames_hq"
    # Assuming audio and subs are flat, adjust if they are also nested by show
    tvqa_audio_dir = tvqa_base_dir / "tvqa_audio"
    tvqa_subs_dir = tvqa_base_dir / "tvqa_subtitles"

    # --- Validate base paths ---
    if not tvqa_hq_frames_dir.is_dir():
        print(f"Error: TVQA HQ frames directory not found at {tvqa_hq_frames_dir}")
        return
    if not tvqa_audio_dir.is_dir():
        print(f"Warning: TVQA audio directory not found at {tvqa_audio_dir}. Subtitle generation will fail if needed.")
    if not tvqa_subs_dir.is_dir():
        print(f"Warning: TVQA subtitles directory not found at {tvqa_subs_dir}. Will rely on audio for generation.")

    # --- Collect all clip paths ---
    clips_to_process = []
    print(f"Scanning for clips in {tvqa_hq_frames_dir}...")
    show_dirs = [d for d in tvqa_hq_frames_dir.iterdir() if d.is_dir()]
    if not show_dirs:
         print(f"Error: No show directories found under {tvqa_hq_frames_dir}")
         return

    for show_dir in show_dirs:
        clip_dirs = [d for d in show_dir.iterdir() if d.is_dir()]
        for clip_dir in clip_dirs:
            clip_name = clip_dir.name
            frame_folder_path = clip_dir
            # Construct paths assuming flat audio/subs directories
            audio_path = tvqa_audio_dir / f"{clip_name}.mp3"
            provided_srt_path = tvqa_subs_dir / f"{clip_name}.srt"
            clips_to_process.append(
                (clip_name, frame_folder_path, audio_path, provided_srt_path)
            )

    print(f"Found {len(clips_to_process)} potential clips across {len(show_dirs)} shows.")

    if not clips_to_process:
        print("No clips found to process.")
        return

    if clip_limit:
        print(f"Processing a limit of {clip_limit} clips.")
        clips_to_process = clips_to_process[:clip_limit]

    # --- Process each collected clip ---
    for clip_name, frame_folder_path, audio_path, provided_srt_path in tqdm(clips_to_process, desc="Processing TVQA Clips"):
        process_single_tvqa_clip(
            clip_name=clip_name,
            frame_folder_path=frame_folder_path, # Pass specific path
            audio_path=audio_path,               # Pass specific path
            provided_srt_path=provided_srt_path, # Pass specific path
            base_output_dir=output_dir,          # Pass base output dir
            num_frames_per_chunk=num_frames_per_chunk,
            skip_frames=skip_frames,
            resize_to=resize_to,
            chunk_weight_function=chunk_weight_function,
            whisper_model=whisper_model,
            force_subtitle_generation=force_subtitle_generation
        )

    print("\nFinished processing TVQA dataset.")


# ==============================================================================
# Example Usage for TVQA Processing
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TVQA dataset frames and audio/subtitles.")
    parser.add_argument("tvqa_dir", type=str, help="Path to the base directory of the extracted TVQA dataset.")
    parser.add_argument("output_dir", type=str, help="Path to the base directory for saving processed output.")
    parser.add_argument("--chunk_size", type=int, default=3, help="Number of frames per chunk (default: 3 for 1 sec @ 3fps).")
    parser.add_argument("--skip", type=int, default=0, help="Number of frames to skip between chunks (default: 0).")
    parser.add_argument("--width", type=int, default=224, help="Resize width.")
    parser.add_argument("--height", type=int, default=224, help="Resize height.")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model size (e.g., tiny, base, small, medium, large).")
    parser.add_argument("--force_generate_subs", action='store_true', help="Force subtitle generation using Whisper even if provided subs exist.")
    parser.add_argument("--limit", type=int, default=3, help="Limit processing to the first N clips (for testing).")
    # Add argument to select averaging function if needed
    # parser.add_argument("--avg_func", type=str, default="blend_blur_with_last_frame", ...)

    args = parser.parse_args()

    # --- Select Averaging Function (Example: using blend_blur_with_last_frame) ---
    # You could make this selectable via args if needed
    averaging_function = blend_blur_with_last_frame

    print("Starting TVQA processing pipeline...")
    print(f"TVQA Base Directory: {args.tvqa_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Chunk Size: {args.chunk_size}, Skip: {args.skip}")
    print(f"Resize To: ({args.width}, {args.height})")
    print(f"Whisper Model: {args.whisper_model}")
    print(f"Force Subtitle Generation: {args.force_generate_subs}")
    if args.limit:
        print(f"Processing Limit: {args.limit} clips")

    process_tvqa_dataset(
        tvqa_base_dir=args.tvqa_dir,
        output_dir=args.output_dir,
        num_frames_per_chunk=args.chunk_size,
        skip_frames=args.skip,
        resize_to=(args.width, args.height),
        chunk_weight_function=averaging_function,
        whisper_model=args.whisper_model,
        force_subtitle_generation=args.force_generate_subs,
        clip_limit=args.limit
    )

    print("\nTVQA processing finished.")
