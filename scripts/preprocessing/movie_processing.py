#!/usr/bin/env python3
"""
Video dataset preprocessing for movie files laid out like:

<data_dir>/
  a_beautiful_mind/
    a_beautiful_mind.mp4
    a_beautiful_mind.srt   # optional but preferred
    annotations/movie.json # optional Q/A with timestamps
  knives_out/
    knives_out.mp4
    knives_out.srt
  ...

This script samples frames from each movie and supports two modes:
  (A) sample by target FPS (e.g., 3 fps) and make chunks of N sampled frames
  (B) sample by wall-clock window length (e.g., 1.0 second per chunk), using native FPS

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

Author: prepared for overnight runs on a single workstation.
"""

import os
import sys
import cv2
import json
import math
import time
import argparse
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm
import pysrt

# -----------------------------------------------------------------------------
# Import your averaging functions
# -----------------------------------------------------------------------------
try:
    from video_preprocessing import (
        weighted_average,
        weighted_average_exponential,
        weighted_average_ramp,
        blend_blur_with_last_frame,
        generate_srt_from_audio,  # optional, only used if you pass --force_generate_subs
    )
except Exception as e:
    print("Warning: Could not import from video_preprocessing.py. Using a fallback simple average.")
    def blend_blur_with_last_frame(chunk: List[np.ndarray]) -> Optional[np.ndarray]:
        if not chunk:
            return None
        arr = np.mean(np.stack(chunk, axis=0), axis=0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def read_subtitles(srt_path: Path) -> List[Tuple[float, float, str]]:
    if not srt_path or not srt_path.exists():
        return []
    try:
        subs = pysrt.open(str(srt_path), encoding='utf-8')
        subs.sort(key=lambda x: x.start)
        parsed = []
        for s in subs:
            start_sec = s.start.ordinal / 1000.0
            end_sec = s.end.ordinal / 1000.0
            text = s.text.replace('\n', ' ').strip()
            if start_sec < end_sec:
                parsed.append((start_sec, end_sec, text))
        return parsed
    except Exception as e:
        print(f"[WARN] Failed to parse SRT {srt_path}: {e}")
        return []


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def natural_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0.1 else 30.0


def sample_step(native_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    step = max(1, int(round(native_fps / target_fps)))
    return step


def read_frame_chunks_from_video(
    video_path: Path,
    target_fps: float,
    chunk_size: int,
    skip_frames_between_chunks: int,
    resize_to: Optional[Tuple[int, int]] = (224, 224),
):
    """Generator of chunks of frames (float32 arrays) sampled at target_fps."""
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
        keep_reading = True
        while keep_reading:
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
                        skipped_sampled = 0
                        while skipped_sampled < to_skip:
                            for _ in range(step - 1):
                                ret2 = cap.grab()
                                if not ret2:
                                    keep_reading = False
                                    break
                            if not keep_reading:
                                break
                            ret2, _ = cap.read()
                            if not ret2:
                                keep_reading = False
                                break
                            skipped_sampled += 1
                            frame_idx += step
                        if not keep_reading:
                            break
                    chunk = []

            frame_idx += 1
    finally:
        cap.release()


def read_chunks_by_seconds(
    video_path: Path,
    seconds_per_chunk: float,
    resize_to: Optional[Tuple[int, int]] = (224, 224),
):
    """Generator: group native frames into ~seconds_per_chunk windows using native FPS.
    Produces contiguous windows with no gaps.
    """
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


def assemble_text_centric_metadata(
    chunk_meta: List[dict],
    subtitle_segments: List[Tuple[float, float, str]],
) -> List[dict]:
    def find_overlaps(t0: float, t1: float):
        out = []
        for info in chunk_meta:
            cs, ce = info['chunk_start_time_sec'], info['chunk_end_time_sec']
            if max(t0, cs) < min(t1, ce):
                out.append({
                    'image_file_range': info['image_file_range'],
                    'saved_chunk_filename': Path(info['saved_image_file']).name,
                    'chunk_start_time_sec': cs,
                    'chunk_end_time_sec': ce,
                })
        return out

    segments = []
    last_end = 0.0
    eps = 1e-3

    for (s, e, txt) in subtitle_segments:
        if s is None or e is None or s >= e:
            continue
        if s > last_end + eps:
            overlaps_gap = find_overlaps(last_end, s)
            if overlaps_gap:
                segments.append({
                    'text': '',
                    'text_start_time_sec': round(last_end, 3),
                    'text_end_time_sec': round(s, 3),
                    'corresponding_chunks': overlaps_gap,
                })
            last_end = s
        overlaps_sub = find_overlaps(s, e)
        segments.append({
            'text': txt,
            'text_start_time_sec': round(s, 3),
            'text_end_time_sec': round(e, 3),
            'corresponding_chunks': overlaps_sub,
        })
        if e > last_end:
            last_end = e

    if chunk_meta:
        last_chunk_end = chunk_meta[-1]['chunk_end_time_sec']
        if last_chunk_end > last_end + eps:
            overlaps_final = find_overlaps(last_end, last_chunk_end)
            if overlaps_final:
                segments.append({
                    'text': '',
                    'text_start_time_sec': round(last_end, 3),
                    'text_end_time_sec': round(last_chunk_end, 3),
                    'corresponding_chunks': overlaps_final,
                })
    return segments


# -----------------------------------------------------------------------------
# Annotation helpers (KEY FRAMES)
# -----------------------------------------------------------------------------

def hhmmss_to_seconds(ts: str) -> Optional[float]:
    try:
        hh, mm, ss = ts.strip().split(':')
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except Exception:
        return None


def load_annotation_items(ann_path: Path) -> List[dict]:
    """Load annotations/movie.json. Accepts either a list or a dict with a top-level key."""
    if not ann_path.exists():
        return []
    try:
        data = json.loads(ann_path.read_text(encoding='utf-8'))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ('items', 'qa', 'annotations', 'data'):
                if k in data and isinstance(data[k], list):
                    return data[k]
        return []
    except Exception as e:
        print(f"[WARN] Failed to read annotations {ann_path}: {e}")
        return []


def extract_key_frame(video_path: Path, t_sec: float, resize_to: Optional[Tuple[int,int]], out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((t_sec+0.001) * natural_fps(cap))))
            ret, frame = cap.read()
            if not ret:
                return False
        if resize_to is not None:
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path), frame)
        return True
    finally:
        cap.release()


# -----------------------------------------------------------------------------
# Per-movie processing
# -----------------------------------------------------------------------------

def process_single_movie(
    movie_dir: Path,
    output_root: Path,
    fps: float,
    chunk_size: int,
    skip_frames: int,
    resize_w: int,
    resize_h: int,
    avg_method: str,
    force_generate_subs: bool,
    whisper_model: str,
    log_every: int = 250,
    chunk_seconds: float = 0.0,
    save_key_frames: bool = True,
    annotations_relpath: str = 'annotations/movie.json',
) -> Tuple[str, bool, str]:
    cv2.setNumThreads(1)

    name = movie_dir.name
    mp4_path = next(movie_dir.glob('*.mp4'), None)
    if mp4_path is None:
        return name, False, "No .mp4 found"
    srt_guess = next(movie_dir.glob('*.srt'), None)

    out_dir = output_root / name
    ensure_dir(out_dir)

    avg_funcs = {
        'blend_blur_with_last_frame': blend_blur_with_last_frame,
        'weighted_average': weighted_average if 'weighted_average' in globals() else blend_blur_with_last_frame,
        'weighted_average_exponential': weighted_average_exponential if 'weighted_average_exponential' in globals() else blend_blur_with_last_frame,
        'weighted_average_ramp': weighted_average_ramp if 'weighted_average_ramp' in globals() else blend_blur_with_last_frame,
    }
    if avg_method not in avg_funcs:
        return name, False, f"Unknown avg_method {avg_method}"
    avg_fn = avg_funcs[avg_method]

    subs: List[Tuple[float, float, str]] = []
    if srt_guess and srt_guess.exists():
        subs = read_subtitles(srt_guess)
    elif force_generate_subs:
        try:
            audio_out = out_dir / f"{name}.wav"
            cmd = f"ffmpeg -y -i '{mp4_path}' -ac 1 -ar 16000 '{audio_out}'"
            rc = os.system(cmd)
            if rc == 0 and 'generate_srt_from_audio' in globals():
                gen_srt_path = out_dir / f"{name}.srt"
                ok = generate_srt_from_audio(str(audio_out), str(gen_srt_path), model_name=whisper_model)
                if ok:
                    subs = read_subtitles(gen_srt_path)
        except Exception:
            pass

    resize_to = (resize_w, resize_h) if (resize_w > 0 and resize_h > 0) else None

    chunk_meta: List[dict] = []
    method_dir = out_dir / avg_method
    ensure_dir(method_dir)

    total_chunks = 0
    frames_processed = 0

    start_time = time.time()
    try:
        if chunk_seconds and chunk_seconds > 0:
            cap_tmp = cv2.VideoCapture(str(mp4_path))
            fps_used = natural_fps(cap_tmp)
            cap_tmp.release()
            gen = read_chunks_by_seconds(mp4_path, seconds_per_chunk=chunk_seconds, resize_to=resize_to)
            eff_chunk_size = 1
            eff_skip = 0
        else:
            fps_used = fps
            gen = read_frame_chunks_from_video(
                mp4_path,
                target_fps=fps,
                chunk_size=chunk_size,
                skip_frames_between_chunks=skip_frames,
                resize_to=resize_to,
            )
            eff_chunk_size = chunk_size
            eff_skip = skip_frames

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
    except Exception as e:
        with open(out_dir / 'logs.txt', 'a', encoding='utf-8') as lf:
            lf.write(f"Exception while processing {name}: {e}\n")
            lf.write(traceback.format_exc() + "\n")
        return name, False, f"Exception: {e}"

    segments = assemble_text_centric_metadata(chunk_meta, subs)
    meta_path = out_dir / 'metadata_text_centric.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    key_ok = 0
    key_total = 0
    if save_key_frames:
        ann_path = movie_dir / annotations_relpath
        items = load_annotation_items(ann_path)
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
        'output_dir': str(out_dir),
    }
    with open(out_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return name, True, f"chunks={total_chunks}, frames={frames_processed}, subs={len(subs)}, key_frames={key_ok}/{key_total}"


# -----------------------------------------------------------------------------
# Discovery & Driver
# -----------------------------------------------------------------------------

def discover_movies(data_dir: Path) -> List[Path]:
    movies = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        if any(d.glob('*.mp4')):
            movies.append(d)
    return movies


def _worker(wi):
    return process_single_movie(*wi)


def main():
    ap = argparse.ArgumentParser(description='Preprocess movie dataset (mp4 + optional srt + optional annotations).')
    ap.add_argument('data_dir', type=str, help='Path to dataset root containing movie folders')
    ap.add_argument('output_dir', type=str, help='Where to write processed outputs')

    # Sampling options
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--fps', type=float, default=3.0, help='Target sampling FPS (default: 3.0)')
    g.add_argument('--chunk_seconds', type=float, default=0.0, help='If >0, ignore --fps/--chunk_size and make one chunk per this many seconds (uses native FPS). For 1.0 you get one blurred frame per second.')

    ap.add_argument('--chunk_size', type=int, default=3, help='Sampled frames per chunk (used only when --chunk_seconds<=0)')
    ap.add_argument('--skip_frames', type=int, default=0, help='Sampled frames to skip BETWEEN chunks (default: 0)')
    ap.add_argument('--width', type=int, default=224, help='Resize width (set <=0 to keep native)')
    ap.add_argument('--height', type=int, default=224, help='Resize height (set <=0 to keep native)')
    ap.add_argument('--avg_method', type=str, default='blend_blur_with_last_frame',
                    choices=['blend_blur_with_last_frame','weighted_average','weighted_average_exponential','weighted_average_ramp'],
                    help='Averaging method to produce one image per chunk')

    # Subtitle options
    ap.add_argument('--force_generate_subs', action='store_true', help='If no .srt, try to generate via Whisper helper (requires ffmpeg)')
    ap.add_argument('--whisper_model', type=str, default='base', help='Whisper model name (if force generating)')

    # Annotation options
    ap.add_argument('--no_key_frames', action='store_true', help='Disable saving KEY_<timestamp>_<index>.jpg from annotations')
    ap.add_argument('--annotations_relpath', type=str, default='annotations/movie.json', help='Relative path from each movie dir to annotations JSON')

    # Parallelism
    ap.add_argument('--workers', type=int, default=max(1, min(4, cpu_count()//2)), help='Parallel movies to process (default ~2-4)')
    ap.add_argument('--limit', type=int, default=None, help='Process at most N movies (debug)')

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    movies = discover_movies(data_dir)
    if args.limit:
        movies = movies[:args.limit]
    if not movies:
        print(f"No movies found under {data_dir} (expecting subfolders each containing an .mp4)")
        sys.exit(1)

    print(f"Discovered {len(movies)} movies. Using {args.workers} workers.")

    worker_inputs = [
        (m, out_root,
         args.fps, args.chunk_size, args.skip_frames,
         args.width, args.height,
         args.avg_method,
         args.force_generate_subs,
         args.whisper_model,
         250,
         args.chunk_seconds,
         not args.no_key_frames,
         args.annotations_relpath)
        for m in movies
    ]

    results = []
    if args.workers == 1:
        for wi in worker_inputs:
            name, ok, msg = process_single_movie(*wi)
            results.append((name, ok, msg))
    else:
        with Pool(processes=args.workers) as pool:
            for name, ok, msg in tqdm(pool.imap_unordered(_worker, worker_inputs), total=len(worker_inputs)):
                results.append((name, ok, msg))

    ok_cnt = sum(1 for _, ok, _ in results if ok)
    print(f"Done. {ok_cnt}/{len(results)} movies processed successfully.")
    for name, ok, msg in results:
        status = 'OK' if ok else 'FAIL'
        print(f" - {name}: {status} — {msg}")


if __name__ == '__main__':
    main()
