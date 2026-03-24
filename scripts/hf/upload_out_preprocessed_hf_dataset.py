#!/usr/bin/env python3
"""Create and upload a staged POVQA dataset folder to the Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, get_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a staged out_preprocessed dataset folder to Hugging Face."
    )
    parser.add_argument(
        "--folder-path",
        default="hf_dataset_staging/povqa_out_preprocessed",
        help="Path to the prepared dataset folder.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo id, e.g. ashimdahal/povqa-out-preprocessed",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision / branch name to upload to.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel workers for upload_large_folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder_path = Path(args.folder_path).resolve()
    if not folder_path.is_dir():
        raise SystemExit(f"Prepared dataset folder not found: {folder_path}")

    readme_path = folder_path / "README.md"
    manifest_path = folder_path / "manifest.json"
    if not readme_path.exists() or not manifest_path.exists():
        raise SystemExit(
            "Expected README.md and manifest.json in the prepared dataset folder."
        )

    token = get_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Run `huggingface-cli login` or set "
            "`HF_TOKEN` before uploading."
        )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        token=token,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=folder_path,
        revision=args.revision,
        private=args.private,
        num_workers=args.num_workers,
        print_report=True,
    )

    print(f"Upload complete: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
