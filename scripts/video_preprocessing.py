import cv2
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
        video_out_dir = output_dir / video_name
        video_out_dir.mkdir(parents=True, exist_ok=True)

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
        output_dir = f"motion_blurred_frames/{weighted_function.__name__}/"
        motion_blur_chunked(
            input_dir="input_videos/",
            output_dir=output_dir,
            num_frames=30,
            skip_frames=0,
            resize_to=(224, 224),
            chunk_weight_function=weighted_function
        )
