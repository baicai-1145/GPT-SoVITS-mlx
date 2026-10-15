"""Unified CLI dispatcher for GPT-SoVITS-mlx."""
from __future__ import annotations

import argparse
import importlib
import sys


COMMANDS = {
    "infer": ("Run text-to-speech inference", "gsovits_mlx.cli.e2e"),
    "e2e": ("Run text-to-speech inference (alias for infer)", "gsovits_mlx.cli.e2e"),
    "download": ("Download pre-converted MLX model weights", "gsovits_mlx.download"),
    "prepare": ("Prepare training dataset (with optional ASR/MSS auto-label)", "gsovits_mlx.cli.prepare_data"),
    "train-s1": ("Train stage-1 AR GPT model", "gsovits_mlx.cli.train_s1"),
    "train-s2": ("Train stage-2 SoVITS GAN model (v1/v2/v2Pro/v2ProPlus)", "gsovits_mlx.cli.train_s2"),
    "train-s2-v3": ("Train stage-2 CFM model (v3/v4/v5dev/v5turbo)", "gsovits_mlx.cli.train_s2_v3"),
    "extract": ("Extract torch weights to training npz layout", "gsovits_mlx.cli.train_s2_extract"),
    "convert": ("Convert PyTorch checkpoint to MLX safetensors", "gsovits_mlx.cli.convert_sovits"),
}


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="gsovits",
        description="GPT-SoVITS on Apple MLX — Inference, Training, and Data Preparation",
    )
    sub = parser.add_subparsers(dest="subcommand", metavar="<command>")

    for name, (help_text, _) in COMMANDS.items():
        sub.add_parser(name, help=help_text, add_help=False)

    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return

    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"Unknown command: '{cmd}'. Choose from: {', '.join(COMMANDS.keys())}\n", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    _, mod_name = COMMANDS[cmd]
    mod = importlib.import_module(mod_name)
    mod.main(argv[1:])


if __name__ == "__main__":
    main()
