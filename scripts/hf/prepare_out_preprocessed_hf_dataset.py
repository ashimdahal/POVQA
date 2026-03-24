#!/usr/bin/env python3
"""Prepare a Hugging Face dataset staging folder from `out_preprocessed`.

The source tree contains hundreds of thousands of files, which is awkward to push
directly to the Hub. This script preserves the movie-centric layout while
packaging each heavy frame directory into a single `.tar` archive.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


DATASET_CARD_TEMPLATE = """---
pretty_name: POVQA Preprocessed Frames and Subtitle-Frame Alignments
language:
- en
task_categories:
- visual-question-answering
tags:
- video
- multimodal
- subtitles
- frame-alignment
---

# POVQA Preprocessed Frames and Subtitle-Frame Alignments

This dataset release contains the **preprocessed artifacts** used by POVQA. It is
intended to support research reproducibility without redistributing raw source
videos or raw subtitle files.

## What is included

- Preprocessed frame bundles for {movie_count} movies and {pooling_count} pooling variants
- `KEY_FRAMES` bundles where available
- Sanitized subtitle-to-frame alignment JSON files
- Per-movie `run_summary.json` metadata
- A `manifest.json` file describing the staged release

## Release Notes

- Total staged size: approximately **{total_size_gb:.2f} GB**
- Total staged files: **{staged_file_count}**
- Selected source size before packaging: **{source_size_gb:.2f} GB**
- Selected source file count before packaging: **{source_file_count}**

## Directory Layout

Each movie has its own subdirectory:

```text
<movie>/
  metadata_text_centric.json
  metadata_text_centric_blend_blur_with_last_frame.json
  metadata_text_centric_weighted_average.json
  metadata_text_centric_weighted_average_exponential.json
  metadata_text_centric_weighted_average_ramp.json
  run_summary.json
  KEY_FRAMES.tar
  blend_blur_with_last_frame.tar
  weighted_average.tar
  weighted_average_exponential.tar
  weighted_average_ramp.tar
```

The `.tar` archives preserve the original folder structure of the preprocessed
release and can be extracted with standard tooling.

## Notes on Data Content

- This release contains **derived, preprocessed artifacts only**.
- Raw videos and raw subtitle files are **not** included in this dataset repo.
- Subtitle credit / torrent source watermark text was scrubbed from the published
  metadata JSON files while preserving timing and frame-mapping fields.

## Intended Use

This release is intended for:

- reproducibility of POVQA preprocessing outputs
- inspection of frame-selection / pooling outputs
- research on video-question-answering pipelines that consume derived frame data

## Responsible Use

Users are responsible for ensuring that their use of this dataset complies with
applicable law, platform terms, and any rights associated with the underlying
source media in their jurisdiction.

## Citation

If you use this dataset, please cite the POVQA paper:

```bibtex
@article{{dahal2025povqa,
  title   = {{POVQA: Preference-Optimized Video Question Answering with Rationales for Data Efficiency}},
  author  = {{Dahal, Ashim and Ghimire, Ankit and Murad, Saydul Akbar and Rahimi, Nick}},
  journal = {{arXiv preprint arXiv:2510.01009}},
  year    = {{2025}},
  url     = {{https://arxiv.org/abs/2510.01009}}
}}
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage out_preprocessed for Hugging Face dataset upload."
    )
    parser.add_argument(
        "--source-dir",
        default="out_preprocessed",
        help="Path to the local preprocessed source tree.",
    )
    parser.add_argument(
        "--output-dir",
        default="hf_dataset_staging/povqa_out_preprocessed",
        help="Where to build the staged dataset folder.",
    )
    parser.add_argument(
        "--movies",
        nargs="*",
        default=None,
        help="Optional subset of movie directory names to stage.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the existing output directory before rebuilding it.",
    )
    return parser.parse_args()


def count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file())


def size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def archive_directory(source_dir: Path, dest_tar: Path) -> tuple[int, int]:
    file_count = count_files(source_dir)
    with tarfile.open(dest_tar, "w") as tar:
        tar.add(source_dir, arcname=source_dir.name)
    archive_size = dest_tar.stat().st_size
    return file_count, archive_size


def copy_metadata_file(source_file: Path, dest_file: Path) -> int:
    shutil.copy2(source_file, dest_file)
    return dest_file.stat().st_size


def build_dataset_card(manifest: dict) -> str:
    return DATASET_CARD_TEMPLATE.format(
        movie_count=manifest["movie_count"],
        pooling_count=manifest["pooling_variant_count"],
        total_size_gb=manifest["staged_bytes"] / (1024**3),
        staged_file_count=manifest["staged_file_count"],
        source_size_gb=manifest["source_bytes"] / (1024**3),
        source_file_count=manifest["source_file_count"],
    )


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {source_dir}")

    if output_dir.exists():
        if not args.force:
            raise SystemExit(
                f"Output directory already exists: {output_dir}\n"
                "Re-run with --force to rebuild it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    movies = sorted([path for path in source_dir.iterdir() if path.is_dir()])
    if args.movies:
        requested = set(args.movies)
        movies = [path for path in movies if path.name in requested]

    if not movies:
        raise SystemExit("No movie directories selected for staging.")

    manifest = {
        "source_dir": str(source_dir),
        "movie_count": len(movies),
        "pooling_variant_count": 4,
        "source_file_count": 0,
        "source_bytes": 0,
        "staged_file_count": 0,
        "staged_bytes": 0,
        "movies": [],
    }

    for movie_dir in movies:
        manifest["source_file_count"] += count_files(movie_dir)
        manifest["source_bytes"] += size_bytes(movie_dir)
        staged_movie_dir = output_dir / movie_dir.name
        staged_movie_dir.mkdir(parents=True, exist_ok=True)

        movie_record = {
            "movie": movie_dir.name,
            "metadata_files": [],
            "archives": [],
        }

        for child in sorted(movie_dir.iterdir()):
            if child.is_file():
                dest = staged_movie_dir / child.name
                copied_size = copy_metadata_file(child, dest)
                manifest["staged_file_count"] += 1
                manifest["staged_bytes"] += copied_size
                movie_record["metadata_files"].append(
                    {"name": child.name, "size_bytes": copied_size}
                )
            elif child.is_dir():
                archive_name = f"{child.name}.tar"
                dest = staged_movie_dir / archive_name
                file_count, archive_size = archive_directory(child, dest)
                manifest["staged_file_count"] += 1
                manifest["staged_bytes"] += archive_size
                movie_record["archives"].append(
                    {
                        "name": archive_name,
                        "source_dir": child.name,
                        "source_file_count": file_count,
                        "size_bytes": archive_size,
                    }
                )

        manifest["movies"].append(movie_record)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["staged_file_count"] += 1
    manifest["staged_bytes"] += manifest_path.stat().st_size

    readme_path = output_dir / "README.md"
    readme_path.write_text(build_dataset_card(manifest))
    manifest["staged_file_count"] += 1
    manifest["staged_bytes"] += readme_path.stat().st_size

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Staged dataset written to: {output_dir}")
    print(f"Movies: {manifest['movie_count']}")
    print(f"Source size: {manifest['source_bytes'] / (1024**3):.2f} GB")
    print(f"Staged size: {manifest['staged_bytes'] / (1024**3):.2f} GB")
    print(f"Staged file count: {manifest['staged_file_count']}")


if __name__ == "__main__":
    main()
