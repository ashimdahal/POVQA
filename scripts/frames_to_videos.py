import os
import cv2
import time
import argparse

from pathlib import Path
import numpy as np 


def load_frames(frame_dir):
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("*.jpg"))
    return [cv2.imread(str(f)) for f in frames if f.exists()]

def resize_to_common(frames, target_size=(224, 224)):
    return [cv2.resize(f, target_size) if f is not None else None for f in frames]


def frame_generator_by_index(frame_dir, target_size=(224, 224)):
    """
    Yields resized frames in order: frame1.jpg, frame2.jpg, ... until a missing frame is hit.
    """
    frame_dir = Path(frame_dir)
    i = 1
    while True:
        frame_path = frame_dir / f"frame{i}.jpg"
        if not frame_path.exists():
            break
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            if target_size:
                frame = cv2.resize(frame, target_size)
            yield frame
        i += 1

def play_comparison(
    base_dir,
    video_name,
    weight_fns,
    fps=1,
    resize_to=(224, 224)
):
    frame_lists = []
    for fn in weight_fns:
        fn_dir = Path(base_dir) / fn / video_name
        # frames = load_frames(fn_dir) # needs frames sorted by index, bit more complicated to implement
        frames = list(frame_generator_by_index(fn_dir, resize_to))
        frame_lists.append(frames)

    # Determine shortest length across all modes
    min_len = min(len(f) for f in frame_lists)

    delay = 1.0 / fps

    for i in range(min_len):
        # Concatenate the i-th frame of each weighting function side by side
        combined = cv2.hconcat([frame_lists[j][i] for j in range(len(weight_fns))])

        # 1) Create an empty “label bar” at the top, same width as 'combined'
        height, width, _ = combined.shape
        label_bar_height = 40
        label_bar = np.zeros((label_bar_height, width, 3), dtype=np.uint8)
        
        # 2) Compute the subregion width for each “column” of frames
        num_fns = len(weight_fns)
        single_width = width // num_fns  # integer division

        # 3) Draw each weighting function name in the correct subregion
        for idx, fn_name in enumerate(weight_fns):
            # Horizontal start/end for this sub-panel
            x_start = idx * single_width
            x_end = (idx + 1) * single_width

            # Choose some offset for the text
            text_x = x_start + 10
            text_y = 25  # about mid of 40 px bar

            cv2.putText(
                label_bar,
                fn_name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,                 # font scale
                (255, 255, 255),     # white text
                1,
                cv2.LINE_AA
            )

        # 4) Stack label_bar above the combined frames
        output_frame = cv2.vconcat([label_bar, combined])

        cv2.imshow("Blur Comparison", output_frame)
        key = cv2.waitKey(int(delay * 1000))
        if key == 27:  # ESC to break
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare motion blur weighting modes side-by-side.")
    parser.add_argument("video_name", type=str, help="Name of subfolder (e.g. 'video1')")
    parser.add_argument("--base_dir", type=str, default="motion_blurred_frames", help="Parent directory of all blur outputs")
    parser.add_argument("--fps", type=float, default=1.0, help="Playback speed")
    args = parser.parse_args()

    weight_fns = os.listdir(args.base_dir)
    play_comparison(args.base_dir, args.video_name, weight_fns, fps=args.fps)
