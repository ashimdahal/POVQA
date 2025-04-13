import os
import cv2
import subprocess
import whisper
import re 
import json
import datetime 

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

def parse_srt_time(time_str):
    """Converts SRT time format HH:MM:SS,ms to seconds."""
    try:
        # Handle potential variations in milliseconds separator (',' or '.')
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            mins = int(parts[1])
            sec_ms = parts[2].split('.')
            secs = int(sec_ms[0])
            ms = int(sec_ms[1].ljust(3, '0')[:3]) if len(sec_ms) > 1 else 0
            total_seconds = hours * 3600 + mins * 60 + secs + ms / 1000.0
            return total_seconds
    except Exception as e:
        print(f"Warning: Could not parse SRT time '{time_str}': {e}")
    return None # Return None on parsing error

def parse_srt_file(srt_path):
    """Parses an SRT file into a list of (start_sec, end_sec, text) tuples."""
    if not srt_path.exists():
        print(f"SRT file not found: {srt_path}")
        return []

    subtitles = []
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Regex to find blocks: index, timestamp, text
            # Allows for multi-line text
            pattern = re.compile(r'(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n(.*?)(?=\n\n|\Z)', re.DOTALL | re.MULTILINE)
            matches = pattern.findall(content)

            for match in matches:
                index, start_str, end_str, text = match
                start_sec = parse_srt_time(start_str)
                end_sec = parse_srt_time(end_str)
                clean_text = text.strip().replace('\n', ' ') # Clean up text

                if start_sec is not None and end_sec is not None:
                    subtitles.append((start_sec, end_sec, clean_text))
                else:
                     print(f"Skipping subtitle index {index} due to timestamp parsing error.")

    except Exception as e:
        print(f"Error reading or parsing SRT file {srt_path}: {e}")
        return [] # Return empty list on error

    # Sort by start time just in case the SRT isn't perfectly ordered
    subtitles.sort(key=lambda x: x[0])
    return subtitles

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

def motion_blur_chunked(
    input_dir,
    output_dir,
    num_frames=30,
    skip_frames=0, # Changed default to 0 for clarity
    resize_to=None,
    chunk_weight_function=weighted_average,
    whisper_model="base" # Added whisper model selection
):
    """
    Processes videos: extracts/generates subtitles, creates motion-blur frames
    per chunk, aligns subtitles, and saves results with metadata.json.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']])
    if not video_paths:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_paths)} videos to process.")

    for video_path in tqdm(video_paths, desc="Processing videos"):
        print(f"\n--- Processing {video_path.name} ---")
        video_name = video_path.stem
        # Base output directory for this specific video
        video_base_out_dir = output_dir / video_name
        video_base_out_dir.mkdir(parents=True, exist_ok=True)

        # Output directory for the specific frame averaging method
        frame_method_name = chunk_weight_function.__name__
        video_frames_out_dir = video_base_out_dir / frame_method_name
        video_frames_out_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Handle Subtitles (Extract or Generate) ---
        subtitle_path = video_base_out_dir / f"{video_name}.srt"
        subtitles_exist = extract_subtitles_if_present(video_path, subtitle_path)

        if not subtitles_exist:
            print("Attempting to generate subtitles with Whisper...")
            generation_successful = generate_srt_with_whisper(str(video_path), str(subtitle_path), model_name=whisper_model)
            if not generation_successful:
                print(f"Warning: Failed to generate subtitles for {video_path.name}. Alignment will be skipped.")
                subtitles_exist = False # Ensure flag is False
            else:
                subtitles_exist = True # Generation succeeded

        # --- 2. Parse Subtitles (if they exist) ---
        parsed_subtitles = []
        if subtitles_exist:
            print(f"Parsing subtitles from: {subtitle_path}")
            parsed_subtitles = parse_srt_file(subtitle_path)
            if not parsed_subtitles:
                print(f"Warning: Subtitle file {subtitle_path} was empty or could not be parsed. No text alignment possible.")
        else:
             print("No subtitles available for alignment.")

        # --- 3. Process Video Frames and Align Text ---
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"⚠️ Could not open video {video_path.name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            print(f"⚠️ Could not get valid FPS for {video_path.name}. Assuming 30 FPS for alignment.")
            fps = 30 # Fallback FPS

        video_metadata = [] # List to store metadata for this video
        total_frames_processed_in_video = 0

        print(f"Processing frames with chunk_size={num_frames}, skip={skip_frames}, method='{frame_method_name}'...")
        # Use tqdm for inner loop progress if desired, though outer loop might be sufficient
        # for i, chunk in tqdm(enumerate(read_video_in_chunks(cap, num_frames, resize_to, skip_frames)), desc="Processing chunks"):
        for i, chunk in enumerate(read_video_in_chunks(cap, num_frames, resize_to, skip_frames)):
            if not chunk: continue # Skip if chunk is empty (shouldn't happen with current logic)

            # --- Calculate Chunk Timestamps ---
            # Frame index marks the start of the *cycle* for this chunk
            start_cycle_frame_index = i * (num_frames + skip_frames)
            # The actual frames used are from start_cycle_frame_index to start_cycle_frame_index + num_frames - 1
            chunk_start_time = start_cycle_frame_index / fps
            # End time is the start of the next frame after the chunk ends
            chunk_end_time = (start_cycle_frame_index + num_frames) / fps
            total_frames_processed_in_video += len(chunk) # Track actual frames used

            # --- Apply Motion Blur ---
            blur = chunk_weight_function(chunk)
            if blur is None: # Handle potential failure in weight function
                print(f"Warning: Frame averaging failed for chunk {i} in {video_path.name}")
                continue

            # --- Save Motion Blur Frame ---
            # Use relative path for metadata
            relative_image_filename = f"{frame_method_name}/frame{i+1:05d}.jpg" # Padded index
            out_path_img = video_base_out_dir / relative_image_filename
            # Ensure the directory exists (needed because relative_image_filename includes subfolder)
            out_path_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path_img), blur)

            # --- Align Subtitles ---
            aligned_text = ""
            if parsed_subtitles:
                overlapping_texts = []
                for sub_start, sub_end, sub_text in parsed_subtitles:
                    # Check for overlap: max(chunk_start, sub_start) < min(chunk_end, sub_end)
                    if max(chunk_start_time, sub_start) < min(chunk_end_time, sub_end):
                        overlapping_texts.append(sub_text)
                aligned_text = " ".join(overlapping_texts).strip()

            # --- Store Metadata ---
            chunk_metadata = {
                "image_file": relative_image_filename, # Relative path within video's output dir
                "start_time_sec": round(chunk_start_time, 3),
                "end_time_sec": round(chunk_end_time, 3),
                "text": aligned_text
            }
            video_metadata.append(chunk_metadata)

        print(f"Processed {total_frames_processed_in_video} frames into {len(video_metadata)} chunks.")
        cap.release()

        # --- 4. Save Metadata to JSON ---
        metadata_path = video_base_out_dir / "metadata.json"
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(video_metadata, f, indent=4, ensure_ascii=False)
            print(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            print(f"Error saving metadata to {metadata_path}: {e}")

        print(f"--- Finished {video_path.name} ---")

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
    print(f"Whisper available: {WHISPER_AVAILABLE}")

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
