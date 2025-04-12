import cv2
import time
from pathlib import Path
import argparse

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
        combined = cv2.hconcat(
            [frame_lists[j][i] for j in range(len(weight_fns))]
        )
        cv2.imshow("Blur Comparison", combined)
        key = cv2.waitKey(int(delay * 1000))
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare motion blur weighting modes side-by-side.")
    parser.add_argument("video_name", type=str, help="Name of subfolder (e.g. 'video1')")
    parser.add_argument("--base_dir", type=str, default="motion_blurred_frames", help="Parent directory of all blur outputs")
    parser.add_argument("--fps", type=float, default=1.0, help="Playback speed")
    args = parser.parse_args()

    weight_fns = [
        "weighted_average",
        "weighted_average_exponential",
        "weighted_average_ramp"
    ]

    play_comparison(args.base_dir, args.video_name, weight_fns, fps=args.fps)
