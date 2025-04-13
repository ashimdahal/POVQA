import cv2
import subprocess
import whisper

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

def motion_blur_chunked(
    input_dir,
    output_dir,
    num_frames=30,
    skip_frames=1,
    resize_to=None,
    chunk_weight_function=weighted_average
):
    """
    Grabs num_frames frames from video, then skips grabbing frames for the
    next skip_frames frames. Every chunk of frames grabbed is then averaged
    using the chunk_weight_function and outputted to the output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(input_dir.glob("*.mp4"))

    for video_path in tqdm(video_paths, desc="Processing videos"):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"⚠️ Could not open {video_path.name}")
            continue

        video_name = video_path.stem
        video_out_dir = output_dir / video_name / chunk_weight_function.__name__
        video_out_dir.mkdir(parents=True, exist_ok=True)

        subtitle_path = output_dir / video_name / "subtitles.srt"
        extract_subtitles_if_present(video_path, subtitle_path)

        for i, chunk in enumerate(read_video_in_chunks(
            cap, num_frames, resize_to, skip_frames
        )):
            blur = chunk_weight_function(chunk)
            out_path = video_out_dir / f"frame{i+1}.jpg"
            cv2.imwrite(str(out_path), blur)

        cap.release()


if __name__ == "__main__":
    weighted_average_functions = [
        weighted_average,
        weighted_average_exponential,
        weighted_average_ramp,
        blend_blur_with_last_frame
    ]

    for weighted_function in weighted_average_functions:
        output_dir = f"motion_blurred_frames/"
        motion_blur_chunked(
            input_dir="input_videos/",
            output_dir=output_dir,
            num_frames=30,
            skip_frames=0,
            resize_to=(224, 224),
            chunk_weight_function=weighted_function
        )
