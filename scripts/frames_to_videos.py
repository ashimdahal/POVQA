import os
import cv2
import time
import argparse
import json # Added for loading metadata
from pathlib import Path
import numpy as np
import itertools # Added for reconstructing generator

# (Helper functions load_frames, resize_to_common are not used in play_comparison, but kept for context)
def load_frames(frame_dir):
    """Loads all .jpg frames from a directory, sorted."""
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("*.jpg"))
    return [cv2.imread(str(f)) for f in frames if f.exists()]

def resize_to_common(frames, target_size=(224, 224)):
    """Resizes a list of frames to a target size."""
    return [cv2.resize(f, target_size, interpolation=cv2.INTER_AREA) if f is not None else None for f in frames]

# --- MODIFIED Function ---
def frame_generator_by_index(frame_dir, target_size=(224, 224)):
    """
    Yields resized frames in order: frame000000001.jpg, frame000000002.jpg, ...
    Uses 9-digit padding to match the provided metadata example.
    """
    frame_dir = Path(frame_dir)
    i = 1
    while True:
        # *** CHANGE: Use 9-digit padding ***
        frame_path = frame_dir / f"frame{i:09d}.jpg"
        if not frame_path.exists():
            # print(f"Debug: Frame not found, stopping: {frame_path}") # Optional debug
            break
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            if target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            yield frame
        else:
            # print(f"Debug: Frame loaded as None: {frame_path}") # Optional debug
            break # Stop if frame fails to load
        i += 1

# --- MODIFIED Function ---
def play_comparison(
    base_dir,
    video_name,
    weight_fns, # List of weight function names (directory names)
    fps=1,
    resize_to=(224, 224)
):
    """
    Plays back comparison frames side-by-side with synchronized subtitles at the bottom.

    Args:
        base_dir (Path): Parent directory of all blur outputs (e.g., 'processed_videos').
        video_name (str): Name of the video subfolder (e.g., 'video1').
        weight_fns (list): List of strings, names of the weight function subdirs to compare.
        fps (float): Playback speed in chunks (output frames) per second.
        resize_to (tuple): Target size (width, height) for each frame panel.
    """
    base_dir = Path(base_dir)
    video_output_dir = base_dir / video_name

    # --- 1. Load Metadata ---
    metadata_path = video_output_dir / "metadata.json"
    metadata_segments = [] # List of text/gap segments
    chunk_times = {} # Map: saved_chunk_filename -> (start_time, end_time)

    if metadata_path.exists():
        print(f"DEBUG: Loading metadata from: {metadata_path}") # DEBUG Print
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_segments = json.load(f)
            print(f"DEBUG: Loaded {len(metadata_segments)} segments.") # DEBUG Print

            # --- Build Chunk Time Lookup Map ---
            for segment in metadata_segments:
                for chunk_data in segment.get("corresponding_chunks", []):
                    filename = chunk_data.get("saved_chunk_filename") # Expecting 9-digit padding here now
                    start_time = chunk_data.get("chunk_start_time_sec")
                    end_time = chunk_data.get("chunk_end_time_sec")
                    if filename and start_time is not None and end_time is not None:
                        chunk_times[filename] = (start_time, end_time)
            print(f"DEBUG: Built time lookup map with {len(chunk_times)} entries.") # DEBUG Print
            # print(f"DEBUG: Example chunk_times entry: {list(chunk_times.items())[:2]}") # DEBUG Print example

        except Exception as e:
            print(f"⚠️ Error loading or parsing metadata {metadata_path}: {e}")
            metadata_segments = []
            chunk_times = {}
    else:
        print(f"⚠️ Metadata file not found: {metadata_path}. Subtitles will not be displayed.")

    # --- 2. Load Frames for Each Method ---
    frame_generators = []
    valid_weight_fns = []
    for fn in weight_fns:
        fn_dir = video_output_dir / fn
        if fn_dir.is_dir():
            # Frame generator now expects 9-digit padding
            gen = frame_generator_by_index(fn_dir, resize_to)
            first_frame = next(gen, None)
            if first_frame is not None:
                frame_generators.append(itertools.chain([first_frame], gen))
                valid_weight_fns.append(fn)
                print(f"Found frames for method: {fn}")
            else:
                 print(f"No frames found or loaded for method: {fn} in {fn_dir}")
        else:
            print(f"Directory not found for method: {fn} in {fn_dir}")

    if not frame_generators:
        print("Error: No valid frame sequences found for any specified weight function. Exiting.")
        return

    # --- 3. Playback Loop ---
    delay = 1.0 / fps
    frame_index = 0 # 0-based index

    while True:
        current_frames = []
        try:
            for gen in frame_generators:
                frame = next(gen)
                current_frames.append(frame)
        except StopIteration:
            print("End of frame sequence reached.")
            break

        if len(current_frames) != len(frame_generators):
            print("Mismatch in frame generator lengths. Stopping.")
            break

        # --- Create Top Label Bar ---
        combined_frames = cv2.hconcat(current_frames)
        height, width, _ = combined_frames.shape
        label_bar_height = 30
        label_bar = np.zeros((label_bar_height, width, 3), dtype=np.uint8)
        num_fns = len(valid_weight_fns)
        single_width = width // num_fns

        for idx, fn_name in enumerate(valid_weight_fns):
            text_x = idx * single_width + 10
            text_y = 20
            cv2.putText(label_bar, fn_name, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Find and Prepare Subtitle ---
        subtitle_text = ""
        # *** CHANGE: Use 9-digit padding ***
        # Construct filename for current chunk (frame_index is 0-based, filenames are 1-based)
        current_chunk_filename = f"frame{frame_index + 1:09d}.jpg"
        # print(f"\nDEBUG: Frame Index: {frame_index}, Expected Filename: {current_chunk_filename}") # DEBUG Print

        # Look up time for this chunk (using the 9-digit filename)
        current_chunk_start_time, current_chunk_end_time = chunk_times.get(current_chunk_filename, (None, None))
        # print(f"DEBUG: Lookup Time: Start={current_chunk_start_time}, End={current_chunk_end_time}") # DEBUG Print

        active_segment_found = False
        if current_chunk_start_time is not None and metadata_segments:
            # Find the active text segment
            for idx, segment in enumerate(metadata_segments):
                seg_start = segment.get("text_start_time_sec")
                seg_end = segment.get("text_end_time_sec")
                if seg_start is not None and seg_end is not None:
                    # Check if chunk start time falls within this segment's duration
                    if seg_start <= current_chunk_start_time < seg_end:
                        subtitle_text = segment.get("text", "")
                        active_segment_found = True
                        break # Found the active segment

        # print(f"DEBUG: Found Subtitle Text: '{subtitle_text}'") # DEBUG Print

        # --- Create Bottom Subtitle Bar ---
        subtitle_bar_height = 40
        subtitle_bar = np.zeros((subtitle_bar_height, width, 3), dtype=np.uint8)
        font_scale = 0.5
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        max_text_width = width - 20

        # Simple text clipping
        text_size, _ = cv2.getTextSize(subtitle_text, font, font_scale, thickness)
        text_w = text_size[0]
        display_text = subtitle_text
        while text_w > max_text_width and len(display_text) > 0:
             display_text = display_text[:-1]
             text_size, _ = cv2.getTextSize(display_text + "...", font, font_scale, thickness)
             text_w = text_size[0]
        if len(display_text) < len(subtitle_text):
            display_text += "..."

        text_x_sub = (width - text_w) // 2
        text_y_sub = subtitle_bar_height - 15
        cv2.putText(subtitle_bar, display_text, (text_x_sub, text_y_sub),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        # print(f"DEBUG: Displaying text: '{display_text}'") # DEBUG Print

        # --- Combine All Parts ---
        output_frame = cv2.vconcat([label_bar, combined_frames, subtitle_bar])

        # --- Display Frame ---
        cv2.imshow("Blur Comparison with Subtitles", output_frame)
        key = cv2.waitKey(int(delay * 1000))
        if key == 27:
            break

        frame_index += 1

    cv2.destroyAllWindows()
    print("Playback finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare motion blur weighting modes side-by-side with subtitles.")
    parser.add_argument("video_name", type=str, help="Name of the processed video subfolder (e.g., 'video1')")
    parser.add_argument("--base_dir", type=str, default="processed_videos", help="Base directory containing processed video folders (default: 'processed_videos')")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback speed in processed frames (chunks) per second (default: 5.0)")
    parser.add_argument("--methods", nargs='*', help="Optional: Specific method subfolders to compare (e.g., weighted_average blend_blur_with_last_frame). If omitted, uses all subfolders found.")
    parser.add_argument("--width", type=int, default=224, help="Width for resizing each frame panel")
    parser.add_argument("--height", type=int, default=224, help="Height for resizing each frame panel")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    video_output_dir = base_dir / args.video_name

    if not video_output_dir.is_dir():
        print(f"Error: Video directory not found: {video_output_dir}")
        exit(1)

    if args.methods:
        weight_fns = args.methods
    else:
        try:
            weight_fns = sorted([d.name for d in video_output_dir.iterdir() if d.is_dir()])
        except OSError as e:
            print(f"Error listing directories in {video_output_dir}: {e}")
            exit(1)

    if not weight_fns:
        print(f"Error: No method subdirectories found in {video_output_dir}")
        exit(1)

    print(f"Attempting to visualize methods: {weight_fns}")

    play_comparison(
        base_dir,
        args.video_name,
        weight_fns,
        fps=args.fps,
        resize_to=(args.width, args.height)
    )
