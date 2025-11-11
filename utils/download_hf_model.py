"""
Utility script to download a Hugging Face model snapshot to a local directory.

Usage:
    python download_hf_model.py <repo_id> <local_dir>

Example:
    python download_hf_model.py TheBlock/Mistral-7B-Instruct-v0.2-GGUF mistral

Note:
Requires:
    pip install huggingface_hub
"""

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download, model_info, HfApi


def get_model_total_size(repo_id: str) -> str:
    total_bytes = 0
    api = HfApi()
    info = api.model_info(repo_id)

    if info.siblings:
        for f in info.siblings:
            if hasattr(f, "size") and f.size:
                total_bytes += f.size

    return f"{total_bytes / (1024**3):.2f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HF repo snapshot to local directory.")
    parser.add_argument("repo_id", help="Hugging Face model repo id, e.g. 'sarvamai/sarvam-1'")
    parser.add_argument(
        "local_dir", help="Local directory to store downloaded files, e.g. './model'"
    )
    args = parser.parse_args()

    local_path = Path(args.local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    try:
        # Fetch metadata
        info = model_info(args.repo_id)
        params = info.safetensors.get("total") if info.safetensors else None
        # total_bytes = info.size or 0
        gb = get_model_total_size(args.repo_id)
        print(f"Model size on HF: ~{gb} ; params: {params}")

        print(f"Downloading '{args.repo_id}' to '{local_path}' ...")
        start = time.time()
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
        )
        elapsed = time.time() - start
        print(f"✅ Download complete in {elapsed:.1f}s.")

        # Print contents summary
        files = list(local_path.rglob("*"))
        print(f"Total files downloaded: {len(files)}")
        print(f"Downloaded {gb} GB model with {params} params in {elapsed:.1f} seconds.")
    # except HfHubHTTPError as e:
    #     print(f"❌ HuggingFace Hub error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
