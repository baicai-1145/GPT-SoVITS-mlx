"""Model weight downloader for GPT-SoVITS-mlx.

Downloads converted MLX model weights from Hugging Face / ModelScope / hf-mirror.
Weights land in GSOVITS_MODELS_ROOT or ~/.cache/gsovits/models by default.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from gsovits_mlx.paths import default_models_root

DEFAULT_REPO = "baicai-1145/GPT-SoVITS-mlx"

# Required file list per component
MODEL_FILES: dict[str, list[str]] = {
    "base": [
        "bert/bert.safetensors",
        "bert/config.json",
        "bert/tokenizer.json",
        "hubert/hubert.safetensors",
        "hubert/config.json",
        "g2pw/g2pw.safetensors",
        "ref_zh_3.5s.wav",
    ],
    "v1": ["v1/sovits.safetensors", "v1/sovits.json", "s1v1/gpt.safetensors", "s1v1/gpt.json"],
    "v2": ["v2/sovits.safetensors", "v2/sovits.json", "s1v2/gpt.safetensors", "s1v2/gpt.json"],
    "v2pro": [
        "v2pro/sovits.safetensors",
        "v2pro/sovits.json",
        "s1v2/gpt.safetensors",
        "s1v2/gpt.json",
        "sv/sv.safetensors",
        "sv/sv.json",
    ],
    "v2proplus": [
        "v2proplus/sovits.safetensors",
        "v2proplus/sovits.json",
        "s1v2/gpt.safetensors",
        "s1v2/gpt.json",
        "sv/sv.safetensors",
        "sv/sv.json",
    ],
    "v3": [
        "v3/sovits.safetensors",
        "v3/sovits.json",
        "s1/gpt.safetensors",
        "s1/gpt.json",
        "bigvgan/bigvgan.safetensors",
        "bigvgan/bigvgan.json",
    ],
    "v4": [
        "v4/sovits.safetensors",
        "v4/sovits.json",
        "s1/gpt.safetensors",
        "s1/gpt.json",
        "v4_vocoder/vocoder.safetensors",
        "v4_vocoder/vocoder.json",
    ],
    "v5dev": [
        "v5dev/sovits.safetensors",
        "v5dev/sovits.json",
        "s1/gpt.safetensors",
        "s1/gpt.json",
        "v5_vocoder/vocoder.safetensors",
        "v5_vocoder/vocoder.json",
    ],
    "v5turbo": [
        "v5turbo/sovits.safetensors",
        "v5turbo/sovits.json",
        "s1/gpt.safetensors",
        "s1/gpt.json",
        "v5_vocoder/vocoder.safetensors",
        "v5_vocoder/vocoder.json",
    ],
}


def resolve_files(targets: list[str]) -> list[str]:
    """Resolve a list of target names into unique relative file paths."""
    files: set[str] = set()
    files.update(MODEL_FILES["base"])
    for t in targets:
        t_lower = t.lower()
        if t_lower == "all":
            for fl in MODEL_FILES.values():
                files.update(fl)
            break
        if t_lower in MODEL_FILES:
            files.update(MODEL_FILES[t_lower])
        else:
            raise ValueError(f"Unknown target: '{t}'. Choices: {list(MODEL_FILES.keys())} or 'all'")
    return sorted(files)


def check_models(dest_dir: str, targets: list[str]) -> tuple[list[str], list[str]]:
    """Check which files are present vs missing in dest_dir."""
    needed = resolve_files(targets)
    present, missing = [], []
    for rel in needed:
        full = os.path.join(dest_dir, rel)
        if os.path.isfile(full) and os.path.getsize(full) > 0:
            present.append(rel)
        else:
            missing.append(rel)
    return present, missing


def download_file(url: str, dest_path: str) -> None:
    """Download a single URL to a file with simple streaming and progress."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "gsovits-mlx/0.1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := resp.read(65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 / total
                mb = downloaded / (1024 * 1024)
                tot_mb = total / (1024 * 1024)
                sys.stdout.write(f"\r  -> {os.path.basename(dest_path)}: {pct:5.1f}% ({mb:.1f}/{tot_mb:.1f} MB)")
                sys.stdout.flush()
    sys.stdout.write("\n")
    os.replace(temp_path, dest_path)


def download_models(
    dest_dir: str,
    targets: list[str],
    *,
    repo: str = DEFAULT_REPO,
    endpoint: str | None = None,
    mirror: bool = False,
    force: bool = False,
) -> None:
    """Download missing model files for the given targets."""
    if not endpoint:
        endpoint = "https://hf-mirror.com" if mirror else os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    endpoint = endpoint.rstrip("/")

    present, missing = check_models(dest_dir, targets)
    to_download = resolve_files(targets) if force else missing

    print(f"Destination: {dest_dir}")
    print(f"Target components: {', '.join(targets)}")
    print(f"Present: {len(present)} files | Need download: {len(to_download)} files")

    if not to_download:
        print("All requested model files are already present!")
        return

    for i, rel in enumerate(to_download, 1):
        url = f"{endpoint}/{repo}/resolve/main/{rel}"
        dst = os.path.join(dest_dir, rel)
        print(f"[{i}/{len(to_download)}] Downloading {rel} from {url}...")
        try:
            download_file(url, dst)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  [ERROR] File {rel} not found in repository {repo} (HTTP 404).")
                print("  If weights are hosted elsewhere, specify --repo or --endpoint.")
            else:
                print(f"  [ERROR] HTTP {e.code}: {e.reason}")
            raise SystemExit(1) from e
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            raise SystemExit(1) from e

    print(f"Successfully downloaded {len(to_download)} files to {dest_dir}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download GPT-SoVITS-mlx weights")
    parser.add_argument("models", nargs="*", default=["v2"],
                        help="Model families to download: v1, v2, v2Pro, v2ProPlus, v3, v4, v5dev, v5turbo, or 'all'")
    parser.add_argument("--dest", default=None,
                        help="Destination directory (default: GSOVITS_MODELS_ROOT or ~/.cache/gsovits/models)")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"HuggingFace repo ID (default: {DEFAULT_REPO})")
    parser.add_argument("--endpoint", default=None,
                        help="Custom HF endpoint URL (default: HF_ENDPOINT env or https://huggingface.co)")
    parser.add_argument("--mirror", action="store_true",
                        help="Use https://hf-mirror.com (recommended in China mainland)")
    parser.add_argument("--check", action="store_true",
                        help="Only check which files are present without downloading")
    parser.add_argument("--force", action="store_true",
                        help="Re-download files even if present locally")
    args = parser.parse_args(argv)

    dest = args.dest or default_models_root()

    if args.check:
        present, missing = check_models(dest, args.models)
        print(f"Destination: {dest}")
        print(f"Present ({len(present)}):")
        for f in present:
            print(f"  + {f}")
        print(f"Missing ({len(missing)}):")
        for f in missing:
            print(f"  - {f}")
        return

    download_models(
        dest,
        args.models,
        repo=args.repo,
        endpoint=args.endpoint,
        mirror=args.mirror,
        force=args.force,
    )


if __name__ == "__main__":
    main()
