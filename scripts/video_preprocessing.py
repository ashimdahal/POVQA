import os
import cv2
import subprocess
import whisper
import json
import datetime 
import pysrt

import moviepy.editor as mpy
import numpy as np
from pathlib import Path
from tqdm import tqdm


#Helper Functions for weighted average of frames over chunks
def weighted_average(chunk):
    weights = np.linspace(0.1, 1.0, len(chunk))
    weighted = np.zeros_like(chunk[0])
    for frame, w in zip(chunk, weights):
        weighted += w * frame
    return (weighted / np.sum(weights)).astype(np.uint8)

def weighted_average_exponential(chunk, gamma=0.85):
    """
    Apply exponential weighting to emphasize recent frames more.
    gamma < 1 → more recent frames get much higher weight
    """
    n = len(chunk)
    weights = np.array([gamma ** (n - i - 1) for i in range(n)])
    weights /= weights.sum()  # normalize

    weighted = np.zeros_like(chunk[0])
    for frame, w in zip(chunk, weights):
        weighted += w * frame
    return weighted.astype(np.uint8)

def weighted_average_ramp(chunk, power=2):
    """
    Applies a linear or quadratic ramp where later frames dominate.
    power=1: linear, power=2: quadratic ramp
    """
    n = len(chunk)
    weights = np.linspace(1, n, n) ** power
    weights /= weights.sum()

    weighted = np.zeros_like(chunk[0])
    for frame, w in zip(chunk, weights):
        weighted += w * frame
    return weighted.astype(np.uint8)

def blend_blur_with_last_frame(chunk, alpha=0.7):
    """
    Compute the weighted average of the entire chunk, 
    then blend the last frame on top with some alpha factor.
    """
    blur = weighted_average_exponential(chunk)  # or your choice of method
    last_frame = chunk[-1]
    
    # alpha blend: out = alpha*blur + (1-alpha)*last_frame
    result = cv2.addWeighted(blur.astype(np.uint8), alpha,
                             last_frame.astype(np.uint8), 1 - alpha, 0)
    return result

#Helper function to extract subtitles from the mp4 if present
def extract_subtitles_if_present(video_path, out_subtitle_path):
    """
    Checks if there's at least one subtitle stream in `video_path`, and if so,
    extracts the first one (0:s:0) to `out_subtitle_path`.

    Args:
        video_path (Path): Path object for the input video file.
        out_subtitle_path (Path): Path object for the output SRT file.

    Returns:
        bool: True if subtitles were successfully extracted or already existed, False otherwise.

    This requires FFmpeg/FFprobe installed and accessible via command line.
    """
    # If we already have a subtitle file, skip re-extracting
    if out_subtitle_path.exists():
        print(f"Subtitle file already exists: {out_subtitle_path}")
        return True

    try:
        # 1) Check for any subtitle streams using ffprobe:
        #    We look at streams of type 's' and see if any exist.
        cmd_probe = [
            "ffprobe", "-v", "error",
            "-select_streams", "s",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(video_path) # Convert Path to string for subprocess
        ]
        result = subprocess.run(cmd_probe, capture_output=True, text=True, check=True, encoding='utf-8')

        # If result.stdout is non-empty, there's at least one subtitle track
        if result.stdout.strip():
            print(f"Subtitle track found in {video_path.name}. Extracting...")
            # 2) Extract the first subtitle track (0:s:0) to SRT
            cmd_extract = [
                "ffmpeg", "-y", # -y overwrites output file without asking
                "-i", str(video_path),
                "-map", "0:s:0", # Map the first subtitle stream
                str(out_subtitle_path)
            ]
            # Use capture_output to get stderr for potential errors, check=True raises error on failure
            extract_result = subprocess.run(cmd_extract, capture_output=True, text=True, check=True, encoding='utf-8')
            print(f"Subtitles extracted successfully to {out_subtitle_path}")
            return True
        else:
            print(f"No embedded subtitle stream found in {video_path.name}.")
            return False

    except FileNotFoundError:
        print("Error: ffmpeg or ffprobe not found. Make sure they are installed and in your PATH.")
        return False
    except subprocess.CalledProcessError as e:
        # Handle cases where ffprobe finds no subtitle stream (returns non-zero exit code sometimes)
        # or ffmpeg fails to extract.
        if "does not contain any stream" in e.stderr:
             print(f"No embedded subtitle stream found in {video_path.name} (ffprobe).")
        elif "Subtitle stream format" in e.stderr:
             print(f"Extraction failed: Unsupported subtitle format in {video_path.name}.")
        else:
            print(f"Error during subtitle extraction process for {video_path.name}:")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during subtitle extraction: {e}")
        return False

def generate_srt_with_whisper(video_path_str, srt_path_str, model_name="base"):
    """
    Generates an SRT subtitle file from a video file using OpenAI's Whisper.

    Args:
        video_path_str (str): Path to the input video file.
        srt_path_str (str): Path to save the output SRT file.
        model_name (str): Name of the Whisper model to use.

    Returns:
        bool: True if subtitle generation was successful, False otherwise.
    """
    print(f"\nAttempting to generate subtitles using Whisper for: {video_path_str}")
    print(f"Using Whisper model: {model_name}")

    audio_path = None # Initialize audio_path
    output_dir = os.path.dirname(srt_path_str)
    os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists

    # --- 1. Extract Audio ---
    try:
        print("Extracting audio from video...")
        video = mpy.VideoFileClip(video_path_str)
        # Use a more specific temp name in case multiple runs happen
        base_name = os.path.splitext(os.path.basename(video_path_str))[0]
        audio_path = os.path.join(output_dir, f"{base_name}_temp_audio.mp3")
        video.audio.write_audiofile(audio_path, codec='mp3')
        video.close() # Close the video file to release resources
        print(f"Audio extracted successfully to: {audio_path}")
    except Exception as e:
        print(f"Error extracting audio: {e}")
        if 'video' in locals() and video:
             video.close() # Ensure file is closed on error
        if audio_path and os.path.exists(audio_path):
             os.remove(audio_path) # Clean up temp audio if extraction failed mid-way
        return False # Indicate failure

    # --- 2. Load Whisper Model ---
    try:
        print("Loading Whisper model...")
        model = whisper.load_model(model_name)
        print("Whisper model loaded.")
    except Exception as e:
        print(f"Error loading Whisper model '{model_name}': {e}")
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path) # Clean up temp audio file
        return False # Indicate failure

    # --- 3. Transcribe Audio ---
    transcription_successful = False
    try:
        print("Transcribing audio... (This may take a while for long videos)")
        # Set fp16=False if you don't have a CUDA-enabled GPU or run into issues
        result = model.transcribe(audio_path, verbose=True, word_timestamps=True, fp16=True)
        print("Transcription complete.")

        # --- 4. Format as SRT ---
        print("Formatting transcription into SRT format...")
        with open(srt_path_str, "w", encoding="utf-8") as srt_file:
            segment_index = 1
            for segment in result.get('segments', []):
                words_in_segment = segment.get('words', [])
                if not words_in_segment: continue

                start_time_s = words_in_segment[0]['start']
                end_time_s = words_in_segment[-1]['end']

                start_ts = str(datetime.timedelta(seconds=start_time_s)).split('.')
                start_ts_formatted = start_ts[0] + ',' + start_ts[1][:3].ljust(3, '0') if len(start_ts) > 1 else start_ts[0] + ',000'

                end_ts = str(datetime.timedelta(seconds=end_time_s)).split('.')
                end_ts_formatted = end_ts[0] + ',' + end_ts[1][:3].ljust(3, '0') if len(end_ts) > 1 else end_ts[0] + ',000'

                segment_text = segment.get('text', '').strip()
                if not segment_text: continue

                srt_file.write(f"{segment_index}\n")
                srt_file.write(f"{start_ts_formatted} --> {end_ts_formatted}\n")
                srt_file.write(f"{segment_text}\n\n")
                segment_index += 1
        
        print(f"SRT file generated successfully: {srt_path_str}")
        transcription_successful = True

    except Exception as e:
        print(f"Error during transcription or SRT formatting: {e}")
        # If SRT file was partially created, maybe remove it? Or leave it for debugging.
        # For now, we'll leave it but indicate failure.
        transcription_successful = False
    finally:
         # --- 5. Clean up temporary audio file ---
         if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"Temporary audio file deleted: {audio_path}")
            except Exception as e:
                print(f"Warning: Could not delete temporary audio file {audio_path}: {e}")

    return transcription_successful

def read_video_in_chunks(cap, chunk_size, resize_to=None, skip_block=0):
    """
    Efficiently read `chunk_size` continuous frames, then skip `skip_block` frames.
    Uses cap.grab() to skip efficiently.
    """
    chunk = []
    total_cycle = chunk_size + skip_block
    frame_idx = 0

    while True:
        if frame_idx % total_cycle < chunk_size:
            ret, frame = cap.read()
            if not ret:
                break
            if resize_to:
                frame = cv2.resize(frame, resize_to)
            chunk.append(frame.astype(np.float32))

            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        else:
            # Skip the frame efficiently
            ret = cap.grab()
            if not ret:
                break

        frame_idx += 1

def handle_subtitles(video_path, video_base_out_dir, video_name, whisper_model):
    """
    Manages subtitle extraction, generation, and parsing for a video.

    Args:
        video_path (Path): Path to the input video file.
        video_base_out_dir (Path): Base output directory for this video.
        video_name (str): Stem name of the video file.
        whisper_model (str): Name of the Whisper model to use for generation.

    Returns:
        list: A list of parsed subtitle tuples (start_sec, end_sec, text),
              or an empty list if subtitles are unavailable or parsing fails.
    """
    subtitle_path = video_base_out_dir / f"{video_name}.srt"
    subtitles_exist = extract_subtitles_if_present(video_path, subtitle_path)

    if not subtitles_exist:
        print("Attempting to generate subtitles with Whisper...")
        generation_successful = generate_srt_with_whisper(str(video_path), str(subtitle_path), model_name=whisper_model)
        if not generation_successful:
            print(f"Warning: Failed to generate subtitles for {video_name}. Alignment will be skipped.")
            return [] # Return empty list if generation fails
        else:
            subtitles_exist = True # Mark as existing after successful generation

    # Parse subtitles if they exist and pysrt is available
    parsed_subtitles = []
    if subtitles_exist: 
        if subtitle_path.exists():
            print(f"Parsing subtitles from: {subtitle_path} using pysrt")
            try:
                subs = pysrt.open(str(subtitle_path), encoding='utf-8')
                for sub in subs:
                    start_sec = sub.start.ordinal / 1000.0
                    end_sec = sub.end.ordinal / 1000.0
                    clean_text = sub.text.replace('\n', ' ').strip()
                    parsed_subtitles.append((start_sec, end_sec, clean_text))
                if not parsed_subtitles:
                     print(f"Warning: pysrt parsed 0 subtitles from {subtitle_path}. Check file content.")
                else:
                     print(f"Successfully parsed {len(parsed_subtitles)} subtitles.")
            except Exception as e:
                print(f"Error parsing SRT file {subtitle_path} with pysrt: {e}")
                parsed_subtitles = [] # Ensure empty on error
        else:
            print(f"Warning: Subtitle file {subtitle_path} not found despite subtitles_exist=True flag.")
            # No file to parse
    elif subtitles_exist: 
         print("Warning: Subtitles exist but pysrt library not installed. Cannot parse for alignment.")
         # Cannot proceed with alignment
    else:
         print("No subtitles available for alignment.")

    return parsed_subtitles

def process_video_chunks(cap, fps, video_base_out_dir,
                         num_frames, skip_frames, resize_to, chunk_weight_function):
    """
    Processes video frame chunks, applies averaging, saves frames, and collects chunk metadata.
    Returns a list of chunk metadata dictionaries. Does NOT perform text alignment.

    Args:
        cap (cv2.VideoCapture): OpenCV video capture object.
        fps (float): Frames per second of the video.
        video_base_out_dir (Path): Base output directory for this video's files.
        num_frames (int): Number of frames per chunk.
        skip_frames (int): Number of frames to skip between chunks.
        resize_to (tuple or None): Target dimensions for resizing frames.
        chunk_weight_function (callable): Function to average frames in a chunk.

    Returns:
        list: A list of metadata dictionaries for each processed chunk.
              Each dict contains: 'image_file_range', 'saved_image_file',
              'chunk_start_time_sec', 'chunk_end_time_sec'.
    """
    chunk_metadata_list = [] # Store results for this method
    total_frames_processed_in_video = 0
    frame_method_name = chunk_weight_function.__name__

    print(f"Processing frames with chunk_size={num_frames}, skip={skip_frames}, method='{frame_method_name}'...")
    for i, chunk in enumerate(read_video_in_chunks(cap, num_frames, resize_to, skip_frames)):
        if not chunk: continue

        # Calculate frame numbers (1-based) and times (0-based seconds)
        start_cycle_frame_index = i * (num_frames + skip_frames) # 0-based index
        start_frame_num = start_cycle_frame_index + 1          # 1-based frame number
        end_frame_num = start_cycle_frame_index + num_frames     # 1-based frame number

        chunk_start_time = start_cycle_frame_index / fps
        chunk_end_time = (start_cycle_frame_index + num_frames) / fps
        total_frames_processed_in_video += len(chunk)

        # Apply Motion Blur/Averaging
        blur = chunk_weight_function(chunk)
        if blur is None:
            print(f"Warning: Frame averaging failed for chunk {i} (starts frame {start_frame_num})")
            continue

        # Save Motion Blur Frame
        output_image_filename = f"frame{i+1:09d}.jpg" # Padded chunk index
        relative_image_path = f"{frame_method_name}/{output_image_filename}"
        out_path_img = video_base_out_dir / relative_image_path
        out_path_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path_img), blur)

        # Create frame range string
        image_file_range_str = f"frame{start_frame_num:05d}-frame{end_frame_num:05d}"

        # Store Chunk Metadata (without text alignment here)
        chunk_metadata = {
            "image_file_range": image_file_range_str,
            "saved_image_file": relative_image_path, # Keep path to actual saved file
            "chunk_start_time_sec": round(chunk_start_time, 3), # Renamed for clarity
            "chunk_end_time_sec": round(chunk_end_time, 3)   # Renamed for clarity
        }
        chunk_metadata_list.append(chunk_metadata)

    print(f"Processed {total_frames_processed_in_video} frames into {len(chunk_metadata_list)} chunks.")
    # Return the list of chunk metadata
    return chunk_metadata_list

def process_single_video(video_path, base_output_dir, num_frames, skip_frames,
                         resize_to, chunk_weight_function, whisper_model):
    """
    Orchestrates the processing pipeline for a single video file.
    MODIFIED: Creates text-centric metadata structure including gaps between subtitles.
    """
    print(f"\n--- Processing {video_path.name} ---")
    video_name = video_path.stem
    video_base_out_dir = base_output_dir / video_name
    video_base_out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Handle Subtitles (Extract/Generate/Parse)
    # Returns list of (sub_start, sub_end, sub_text), sorted by start time
    parsed_subtitles = handle_subtitles(video_path, video_base_out_dir, video_name, whisper_model)

    # Step 2: Process Video Frames (Get chunk metadata list)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Could not open video {video_path.name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        print(f"⚠️ Could not get valid FPS for {video_path.name}. Assuming 30 FPS.")
        fps = 30.0

    # Get the list of processed chunk metadata, sorted by time
    # Each dict: {'image_file_range', 'saved_image_file', 'chunk_start_time_sec', 'chunk_end_time_sec'}
    chunk_metadata_list = process_video_chunks(
        cap, fps, video_base_out_dir,
        num_frames, skip_frames, resize_to, chunk_weight_function
    )

    cap.release()

    # Step 3: Assemble Text-Centric Metadata Structure including Gaps
    final_metadata_structure = []
    last_event_end_time = 0.0 # Track the end time of the last processed segment (sub or gap)

    if not chunk_metadata_list:
        print("Skipping metadata assembly: No video chunks were processed.")
    else:
        print("Assembling text-centric metadata including gaps...")
        # Helper function to find overlapping chunks for a given time interval
        def find_overlapping_chunks(interval_start, interval_end):
            simplified_chunks = []
            for chunk_info in chunk_metadata_list:
                chunk_start = chunk_info["chunk_start_time_sec"]
                chunk_end = chunk_info["chunk_end_time_sec"]
                # Overlap check: max(starts) < min(ends)
                # Add small tolerance for floating point? Maybe not needed yet.
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
            # Ensure start/end times are valid
            if sub_start is None or sub_end is None or sub_start >= sub_end:
                print(f"Skipping invalid subtitle segment: start={sub_start}, end={sub_end}, text='{sub_text[:20]}...'")
                continue

            # --- Process Gap Before Current Subtitle ---
            # Check if there's a gap between the last event and this subtitle
            # Use a small epsilon to avoid tiny gaps due to float precision
            epsilon = 1e-3
            if sub_start > last_event_end_time + epsilon:
                gap_start = last_event_end_time
                gap_end = sub_start
                corresponding_chunks_gap = find_overlapping_chunks(gap_start, gap_end)
                # Only add gap if it overlaps with some processed chunks
                if corresponding_chunks_gap:
                    final_metadata_structure.append({
                        "text": "", # Empty text for gap
                        "text_start_time_sec": round(gap_start, 3),
                        "text_end_time_sec": round(gap_end, 3),
                        "corresponding_chunks": corresponding_chunks_gap
                    })
                # Update last event end time even if gap wasn't added (to avoid reprocessing)
                last_event_end_time = gap_end # End of the gap period

            # --- Process Current Subtitle Segment ---
            corresponding_chunks_sub = find_overlapping_chunks(sub_start, sub_end)
            # Add subtitle entry (always add, even if no chunks overlap, unless text is empty?)
            # Let's add it always, as it came from the SRT.
            final_metadata_structure.append({
                "text": sub_text,
                "text_start_time_sec": round(sub_start, 3),
                "text_end_time_sec": round(sub_end, 3),
                "corresponding_chunks": corresponding_chunks_sub
            })

            # Update the end time marker
            last_event_end_time = max(last_event_end_time, sub_end) # Use max in case of overlapping subs

        # --- Process Gap After Last Subtitle ---
        # Find the end time of the very last processed chunk
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


    # Step 4: Save Final Metadata Structure to JSON
    if final_metadata_structure:
        metadata_path = video_base_out_dir / "metadata.json" # New filename
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(final_metadata_structure, f, indent=4, ensure_ascii=False)
            print(f"Full coverage text-centric metadata saved to: {metadata_path}")
        except Exception as e:
            print(f"Error saving full coverage metadata to {metadata_path}: {e}")
    else:
        print(f"No text-centric metadata generated for {video_path.name}.")

    print(f"--- Finished {video_path.name} ---")

def motion_blur_chunked(
    input_dir,
    output_dir,
    num_frames=30,
    skip_frames=0,
    resize_to=None,
    chunk_weight_function=weighted_average,
    whisper_model="base"
):
    """
    Iterates through videos in input_dir and processes each one.

    Args:
        input_dir (str or Path): Directory containing input video files.
        output_dir (str or Path): Base directory to save processed output.
        num_frames (int): Number of frames per chunk.
        skip_frames (int): Number of frames to skip between chunks.
        resize_to (tuple or None): Target dimensions for resizing frames (width, height).
        chunk_weight_function (callable): Function to average frames in a chunk.
        whisper_model (str): Whisper model name for subtitle generation.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure base output dir exists

    # Find video files
    video_paths = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']])
    if not video_paths:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_paths)} videos to process in {input_dir.resolve()}.")

    # Process each video using the new helper function
    for video_path in tqdm(video_paths, desc="Processing videos"):
        process_single_video(
            video_path,
            output_dir, # Pass the base output dir
            num_frames,
            skip_frames,
            resize_to,
            chunk_weight_function,
            whisper_model
        )

if __name__ == "__main__":
    print("Starting video processing pipeline...")

    # --- Configuration ---
    INPUT_VIDEO_DIR = "input_videos/"
    BASE_OUTPUT_DIR = "processed_videos/" # Base directory for all output
    FRAMES_PER_CHUNK = 30
    FRAMES_TO_SKIP = 0 # Number of frames to skip between chunks
    RESIZE_DIMENSIONS = (224, 224) # Tuple (width, height), or None to disable
    WHISPER_MODEL_SIZE = "base" # tiny, base, small, medium, large

    # List of frame averaging functions to test
    averaging_functions_to_run = [
        weighted_average,
        weighted_average_exponential,
        weighted_average_ramp,
        blend_blur_with_last_frame
    ]

    # --- Ensure input directory exists ---
    Path(INPUT_VIDEO_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Input directory: {Path(INPUT_VIDEO_DIR).resolve()}")
    print(f"Base output directory: {Path(BASE_OUTPUT_DIR).resolve()}")

    # --- Check for FFmpeg/FFprobe early ---
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        print("FFmpeg and FFprobe found.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\n" + "="*60)
        print("⚠️ WARNING: FFmpeg or FFprobe not found or not working correctly.")
        print("Please ensure they are installed and accessible in your system's PATH.")
        print("Subtitle extraction will fail without them.")
        print("="*60 + "\n")
        exit(1)

    # --- Run processing for each selected averaging function ---
    for avg_func in averaging_functions_to_run:
        print(f"\n=== Running with averaging function: {avg_func.__name__} ===")
        # Output dir will include video name and function name automatically inside the function
        motion_blur_chunked(
            input_dir=INPUT_VIDEO_DIR,
            output_dir=BASE_OUTPUT_DIR, # Pass the base output dir
            num_frames=FRAMES_PER_CHUNK,
            skip_frames=FRAMES_TO_SKIP,
            resize_to=RESIZE_DIMENSIONS,
            chunk_weight_function=avg_func,
            whisper_model=WHISPER_MODEL_SIZE
        )

    print("\nVideo processing pipeline finished.")
