#!/usr/bin/env python3
"""
Video dataset preprocessing for movie files laid out like:

  Example layout:
    data/
      a_beautiful_mind/a_beautiful_mind.mp4
      ...
    annotations/
      a_beautiful_mind.json
      all_quiet.json
      ...

- Accepts --annotations_dir pointing to a single folder that contains one JSON per movie.
- Robust annotation file resolver that matches by normalized movie name to JSON stem
  (case-insensitive, ignores punctuation/underscores). Also tries prefix matches.
- Still ignores macOS resource-fork files (._*) and tiny files.
- SRT parsing falls back to latin-1 if utf-8 fails.
- Supports one blurred frame per second via --chunk_seconds 1.0
- Extracts KEY frames from annotations: KEY_<HH-MM-SS>_<index>.jpg

It also extracts KEY FRAMES from annotations/movie.json entries by seeking to
`timestamp` and saving a JPEG per QA:
  KEY_<HH-MM-SS>_<index>.jpg

Runs multi-process: one process per movie.

Output structure:
<output_dir>/<movie_name>/
  <avg_method>/frame00001.jpg
  KEY_FRAMES/KEY_00-08-32_0000.jpg
  metadata_text_centric.json
  run_summary.json
  logs.txt


Run (one frame per second + KEY frames):
  python scripts/preprocessing/movie_preprocess.py \
    data out_preprocessed \
    --chunk_seconds 1.0 \
    --width 224 --height 224 \
    --avg_method blend_blur_with_last_frame \
    --annotations_dir annotations \
    --workers 3
"""
import os
import sys
import re
import cv2
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

try:
    import pysrt
except Exception:
    pysrt = None

# --- averaging fns ---
try:
    from video_preprocessing import (
        weighted_average,
        weighted_average_exponential,
        weighted_average_ramp,
        blend_blur_with_last_frame,
        generate_srt_from_audio,
    )
except Exception:
    print("Warning: Could not import from video_preprocessing.py. Using a fallback simple average.")
    sys.exit()
    def blend_blur_with_last_frame(chunk: List[np.ndarray]) -> Optional[np.ndarray]:
        if not chunk:
            return None
        arr = np.mean(np.stack(chunk, axis=0), axis=0)
        return np.clip(arr, 0, 255).astype(np.uint8)

# ---------------- helpers ----------------

ALL_METHODS = [
    'blend_blur_with_last_frame',
    'weighted_average',
    'weighted_average_exponential',
    'weighted_average_ramp',
]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_hidden_or_resource(p: Path) -> bool:
    name = p.name
    return name.startswith('.') or name.startswith('._')


def choose_primary_file(folder: Path, patterns: List[str]) -> Optional[Path]:
    """Return the largest non-hidden, non-zero file matching any pattern."""
    cands: List[Path] = []
    for pat in patterns:
        cands.extend(folder.glob(pat))
    cands = [c for c in cands if not is_hidden_or_resource(c) and c.is_file() and c.stat().st_size > 1024]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    return cands[0]


def natural_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0.1 else 30.0


def read_srt_with_fallback(srt_path: Path) -> List[Tuple[float, float, str]]:
    if pysrt is None:
        return []
    if not srt_path or not srt_path.exists():
        return []
    for enc in ('utf-8', 'latin-1'):
        try:
            subs = pysrt.open(str(srt_path), encoding=enc)
            subs.sort(key=lambda x: x.start)
            out = []
            for s in subs:
                start_sec = s.start.ordinal / 1000.0
                end_sec = s.end.ordinal / 1000.0
                txt = s.text.replace('\n', ' ').strip()
                if start_sec < end_sec:
                    out.append((start_sec, end_sec, txt))
            return out
        except Exception:
            continue
    return []


def sample_step(native_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    return max(1, int(round(native_fps / target_fps)))


def read_frame_chunks_from_video(video_path: Path, target_fps: float, chunk_size: int, skip_frames_between_chunks: int, resize_to: Optional[Tuple[int, int]]):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERR] Cannot open video: {video_path}")
        return
    try:
        nat_fps = natural_fps(cap)
        step = sample_step(nat_fps, target_fps)
        chunk = []
        frame_idx = 0
        keep = True
        while keep:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                if resize_to is not None:
                    frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
                chunk.append(frame.astype(np.float32))
                if len(chunk) == chunk_size:
                    yield chunk
                    # skip sampled frames between chunks
                    to_skip = skip_frames_between_chunks
                    if to_skip > 0:
                        skipped = 0
                        while skipped < to_skip:
                            for _ in range(step - 1):
                                if not cap.grab():
                                    keep = False
                                    break
                            if not keep:
                                break
                            ret2, _ = cap.read()
                            if not ret2:
                                keep = False
                                break
                            skipped += 1
                            frame_idx += step
                        if not keep:
                            break
                    chunk = []
            frame_idx += 1
    finally:
        cap.release()


def read_chunks_by_seconds(video_path: Path, seconds_per_chunk: float, resize_to: Optional[Tuple[int, int]]):
    if seconds_per_chunk <= 0:
        raise ValueError("seconds_per_chunk must be positive")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERR] Cannot open video: {video_path}")
        return
    try:
        nat_fps = natural_fps(cap)
        frames_per_chunk = max(1, int(round(nat_fps * seconds_per_chunk)))
        chunk = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resize_to is not None:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            chunk.append(frame.astype(np.float32))
            if len(chunk) == frames_per_chunk:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
    finally:
        cap.release()


def assemble_text_centric_metadata(chunk_meta: List[dict], subtitle_segments: List[Tuple[float, float, str]]) -> List[dict]:
    def overlaps(t0, t1, info):
        return max(t0, info['chunk_start_time_sec']) < min(t1, info['chunk_end_time_sec'])
    out = []
    last_end = 0.0
    eps = 1e-3
    for (s, e, txt) in subtitle_segments:
        if s is None or e is None or s >= e:
            continue
        if s > last_end + eps:
            ch = [
                {
                    'image_file_range': i['image_file_range'],
                    'saved_chunk_filename': Path(i['saved_image_file']).name,
                    'chunk_start_time_sec': i['chunk_start_time_sec'],
                    'chunk_end_time_sec': i['chunk_end_time_sec'],
                }
                for i in chunk_meta if overlaps(last_end, s, i)
            ]
            if ch:
                out.append({'text': '', 'text_start_time_sec': round(last_end,3), 'text_end_time_sec': round(s,3), 'corresponding_chunks': ch})
            last_end = s
        ch2 = [
            {
                'image_file_range': i['image_file_range'],
                'saved_chunk_filename': Path(i['saved_image_file']).name,
                'chunk_start_time_sec': i['chunk_start_time_sec'],
                'chunk_end_time_sec': i['chunk_end_time_sec'],
            }
            for i in chunk_meta if overlaps(s, e, i)
        ]
        out.append({'text': txt, 'text_start_time_sec': round(s,3), 'text_end_time_sec': round(e,3), 'corresponding_chunks': ch2})
        last_end = max(last_end, e)
    if chunk_meta:
        last_chunk_end = chunk_meta[-1]['chunk_end_time_sec']
        if last_chunk_end > last_end + eps:
            ch3 = [
                {
                    'image_file_range': i['image_file_range'],
                    'saved_chunk_filename': Path(i['saved_image_file']).name,
                    'chunk_start_time_sec': i['chunk_start_time_sec'],
                    'chunk_end_time_sec': i['chunk_end_time_sec'],
                }
                for i in chunk_meta if overlaps(last_end, last_chunk_end, i)
            ]
            if ch3:
                out.append({'text': '', 'text_start_time_sec': round(last_end,3), 'text_end_time_sec': round(last_chunk_end,3), 'corresponding_chunks': ch3})
    return out

# --- annotations (KEY frames) ---

def hhmmss_to_seconds(ts: str) -> Optional[float]:
    try:
        hh, mm, ss = ts.strip().split(':')
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except Exception:
        return None


def load_annotation_items(ann_path: Path) -> List[dict]:
    if not ann_path.exists():
        return []
    try:
        data = json.loads(ann_path.read_text(encoding='utf-8'))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ('items','qa','annotations','data'):
                if k in data and isinstance(data[k], list):
                    return data[k]
        return []
    except Exception:
        return []


def normalize_key(s: str) -> str:
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', '', s)
    return s


def locate_annotation_for_movie(annotations_dir: Path, movie_name: str) -> Optional[Path]:
    """Find <annotations_dir>/<movie>.json in a tolerant way.
    1) exact stem match (normalized)
    2) prefix match (normalized)
    3) first .json whose normalized stem shares all tokens of movie_name
    """
    if not annotations_dir or not annotations_dir.exists():
        return None
    norm_movie = normalize_key(movie_name)
    jsons = [p for p in annotations_dir.glob('*.json') if not is_hidden_or_resource(p)]
    # exact match
    for p in jsons:
        if normalize_key(p.stem) == norm_movie:
            return p
    # prefix match
    for p in jsons:
        if normalize_key(p.stem).startswith(norm_movie) or norm_movie.startswith(normalize_key(p.stem)):
            return p
    # token containment (very light-weight)
    # (Not strict; just fallback to the first one that shares a long prefix chunk)
    for p in jsons:
        if any(normalize_key(tok) and normalize_key(tok) in normalize_key(p.stem) for tok in movie_name.replace('_',' ').split()):
            return p
    return None


def extract_key_frame(video_path: Path, t_sec: float, resize_to: Optional[Tuple[int,int]], out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
        ret, frame = cap.read()
        if not ret:
            fps = natural_fps(cap)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((t_sec+0.001) * fps)))
            ret, frame = cap.read()
            if not ret:
                return False
        if resize_to is not None:
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path), frame)
        return True
    finally:
        cap.release()

# -------------- per-movie --------------

def process_single_movie(movie_dir: Path, output_root: Path, fps: float, chunk_size: int, skip_frames: int, resize_w: int, resize_h: int, avg_method: str, force_generate_subs: bool, whisper_model: str, log_every: int = 250, chunk_seconds: float = 0.0, save_key_frames: bool = True, annotations_dir: Optional[Path] = None) -> Tuple[str, bool, str]:
    cv2.setNumThreads(1)
    name = movie_dir.name

    mp4_path = choose_primary_file(movie_dir, ['*.mp4', '*.MP4', '*.mkv', '*.MKV'])
    if mp4_path is None:
        return name, False, "No usable video file found (skipped hidden/._/tiny files)"

    srt_path = choose_primary_file(movie_dir, ['*.srt','*.SRT'])

    out_dir = output_root / name
    ensure_dir(out_dir)

    avg_funcs = {
        'blend_blur_with_last_frame': blend_blur_with_last_frame,
        'weighted_average': globals().get('weighted_average', blend_blur_with_last_frame),
        'weighted_average_exponential': globals().get('weighted_average_exponential', blend_blur_with_last_frame),
        'weighted_average_ramp': globals().get('weighted_average_ramp', blend_blur_with_last_frame),
    }
    if avg_method not in avg_funcs:
        return name, False, f"Unknown avg_method {avg_method}"
    avg_fn = avg_funcs[avg_method]

    subs: List[Tuple[float, float, str]] = []
    if srt_path:
        subs = read_srt_with_fallback(srt_path)

    resize_to = (resize_w, resize_h) if (resize_w > 0 and resize_h > 0) else None

    chunk_meta: List[dict] = []
    method_dir = out_dir / avg_method
    ensure_dir(method_dir)

    total_chunks = 0
    frames_processed = 0

    # choose generator
    if chunk_seconds and chunk_seconds > 0:
        cap_tmp = cv2.VideoCapture(str(mp4_path))
        fps_used = natural_fps(cap_tmp)
        cap_tmp.release()
        gen = read_chunks_by_seconds(mp4_path, seconds_per_chunk=chunk_seconds, resize_to=resize_to)
        eff_chunk_size = 1
        eff_skip = 0
    else:
        fps_used = fps
        gen = read_frame_chunks_from_video(mp4_path, target_fps=fps, chunk_size=chunk_size, skip_frames_between_chunks=skip_frames, resize_to=resize_to)
        eff_chunk_size = chunk_size
        eff_skip = skip_frames

    start_time = time.time()
    for i, chunk in enumerate(gen):
        frames_processed += len(chunk)
        start_cycle_index = i * (eff_chunk_size + eff_skip)
        if chunk_seconds and chunk_seconds > 0:
            chunk_start_t = i * chunk_seconds
            chunk_end_t = chunk_start_t + chunk_seconds
        else:
            chunk_start_t = start_cycle_index / max(1e-6, fps_used)
            chunk_end_t = (start_cycle_index + eff_chunk_size) / max(1e-6, fps_used)

        img = avg_fn(chunk)
        if img is None:
            continue
        out_name = f"frame{i+1:05d}.jpg"
        out_path = method_dir / out_name
        cv2.imwrite(str(out_path), img)

        chunk_meta.append({
            'image_file_range': f"sample{start_cycle_index+1:05d}-sample{start_cycle_index+len(chunk):05d}",
            'saved_image_file': f"{avg_method}/{out_name}",
            'chunk_start_time_sec': round(chunk_start_t, 3),
            'chunk_end_time_sec': round(chunk_end_t, 3),
        })

        total_chunks += 1
        if log_every and (i+1) % log_every == 0:
            elapsed = time.time() - start_time
            print(f"[{name}] Chunks: {i+1}, Frames: {frames_processed}, Elapsed: {elapsed/60:.1f} min")

    # metadata
    segments = assemble_text_centric_metadata(chunk_meta, subs)
    (out_dir / f'metadata_text_centric_{avg_method}.json').write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding='utf-8')

    # key frames (use external annotations dir)
    key_ok = 0
    key_total = 0
    if save_key_frames and annotations_dir:
        ann_path = locate_annotation_for_movie(annotations_dir, name)
        if ann_path and ann_path.exists():
            try:
                items = load_annotation_items(ann_path)
            except Exception:
                items = []
            if items:
                key_dir = out_dir / 'KEY_FRAMES'
                ensure_dir(key_dir)
                for item in items:
                    ts = item.get('timestamp')
                    idx = item.get('index')
                    if ts is None or idx is None:
                        continue
                    t_sec = hhmmss_to_seconds(ts)
                    if t_sec is None:
                        continue
                    safe_ts = ts.replace(':', '-')
                    out_key = key_dir / f"KEY_{safe_ts}_{int(idx):04d}.jpg"
                    if extract_key_frame(mp4_path, t_sec, resize_to, out_key):
                        key_ok += 1
                    key_total += 1

    # summary
    summary = {
        'movie': name,
        'video_path': str(mp4_path),
        'fps_mode': 'seconds_window' if (chunk_seconds and chunk_seconds > 0) else 'target_fps',
        'fps_used_or_native': float(fps_used),
        'target_fps_arg': float(fps),
        'chunk_size_arg': int(chunk_size),
        'chunk_seconds': float(chunk_seconds),
        'skip_frames_between_chunks': int(skip_frames),
        'resize_to': [resize_w, resize_h] if resize_to else None,
        'avg_method': avg_method,
        'total_chunks': total_chunks,
        'frames_processed': frames_processed,
        'has_subtitles': bool(subs),
        'key_frames_found': key_ok,
        'key_frames_total_in_annotations': key_total,
        'annotation_file': str(ann_path) if save_key_frames and annotations_dir and 'ann_path' in locals() and ann_path else None,
        'output_dir': str(out_dir),
    }
    (out_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    return name, True, f"chunks={total_chunks}, frames={frames_processed}, subs={len(subs)}, key_frames={key_ok}/{key_total}"

# -------------- driver --------------

def discover_movies(data_dir: Path) -> List[Path]:
    out = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        if choose_primary_file(d, ['*.mp4','*.MP4','*.mkv','*.MKV']):
            out.append(d)
    return out


def _worker(wi):
    return process_single_movie(*wi)


def main():
    ap = argparse.ArgumentParser(description='Preprocess movie dataset (mp4 + optional srt + external annotations).')
    ap.add_argument('data_dir', type=str)
    ap.add_argument('output_dir', type=str)

    g = ap.add_mutually_exclusive_group()
    g.add_argument('--fps', type=float, default=3.0)
    g.add_argument('--chunk_seconds', type=float, default=0.0)

    ap.add_argument('--chunk_size', type=int, default=3)
    ap.add_argument('--skip_frames', type=int, default=0)
    ap.add_argument('--width', type=int, default=224)
    ap.add_argument('--height', type=int, default=224)

    ap.add_argument(
        '--avg_method',
        type=str,
        default='blend_blur_with_last_frame',
        choices=['blend_blur_with_last_frame','weighted_average','weighted_average_exponential','weighted_average_ramp','all']
    )

    ap.add_argument('--force_generate_subs', action='store_true')
    ap.add_argument('--whisper_model', type=str, default='base')

    ap.add_argument('--no_key_frames', action='store_true')
    ap.add_argument('--annotations_dir', type=str, default=None, help='Folder containing <movie>.json files')

    ap.add_argument('--workers', type=int, default=max(1, min(4, cpu_count()//2)))
    ap.add_argument('--limit', type=int, default=None)

    args = ap.parse_args()


    data_dir = Path(args.data_dir)
    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    movies = discover_movies(data_dir)
    if args.limit:
        movies = movies[:args.limit]
    if not movies:
        print(f"No movies found under {data_dir}")
        sys.exit(1)

    print(f"Discovered {len(movies)} movies. Using {args.workers} workers.")

    ann_dir = Path(args.annotations_dir) if args.annotations_dir else None

    methods_to_run = ALL_METHODS if args.avg_method == 'all' else [args.avg_method]

    worker_inputs = []
    for mi, method in enumerate(methods_to_run):
        # only save KEY frames once (first method) to avoid duplicates
        save_keys = (mi == 0) and (not args.no_key_frames)
        worker_inputs.extend([
            (m, out_root,
             args.fps, args.chunk_size, args.skip_frames,
             args.width, args.height,
             method,                     # <- this method
             args.force_generate_subs,
             args.whisper_model,
             250,
             args.chunk_seconds,
             save_keys,                  # <- only first method extracts KEY frames
             Path(args.annotations_dir) if args.annotations_dir else None)
            for m in movies
        ])

    results = []
    if args.workers == 1:
        for wi in worker_inputs:
            results.append(process_single_movie(*wi))
    else:
        with Pool(processes=args.workers) as pool:
            for name, ok, msg in tqdm(pool.imap_unordered(_worker, worker_inputs), total=len(worker_inputs)):
                results.append((name, ok, msg))

    ok_cnt = sum(1 for _, ok, _ in results if ok)
    print(f"Done. {ok_cnt}/{len(results)} movies processed successfully.")
    for name, ok, msg in results:
        print(f" - {name}: {'OK' if ok else 'FAIL'} — {msg}")

if __name__ == '__main__':
    main()
