#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL_OUTPUTS = {
    "cnn1d": "online_detector_cnn1d_rfly",
    "bilstm": "online_detector_bilstm_rfly",
    "transformer": "online_detector_transformer_rfly",
    "mstca": "online_detector_mstca_rfly",
    "kg_mstca": "online_detector_kg_mstca_rfly",
}


def build_commands(
    *,
    python_bin: str,
    repo_root: Path,
    cache_dir: Path,
    output_root: Path,
    summary_dir: Path,
    models: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_size: int,
    dropout: float,
    mode: str = "",
    num_layers_by_model: dict[str, int] | None = None,
) -> list[list[str]]:
    num_layers_by_model = dict(num_layers_by_model or {})
    train_script = repo_root / "packages" / "uav_fault" / "models" / "train_online_detector.py"
    compare_script = repo_root / "scripts" / "compare_online_backbones.py"

    commands: list[list[str]] = []
    for model in models:
        output_dir = output_root / DEFAULT_MODEL_OUTPUTS[model]
        num_layers = int(num_layers_by_model.get(model, 1))
        commands.append(
            [
                python_bin,
                str(train_script),
                "--model",
                model,
                "--cache-dir",
                str(cache_dir),
                "--output-dir",
                str(output_dir),
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--lr",
                str(lr),
                "--hidden-size",
                str(hidden_size),
                "--dropout",
                str(dropout),
                "--num-layers",
                str(num_layers),
            ]
        )
        if mode:
            commands[-1].extend(["--mode", mode])

    compare_cmd = [
        python_bin,
        str(compare_script),
        "--results-root",
        str(output_root),
        "--output-dir",
        str(summary_dir),
    ]
    for model in models:
        compare_cmd.extend(
            [
                "--experiment",
                f"{model}={DEFAULT_MODEL_OUTPUTS[model]}",
            ]
        )
    commands.append(compare_cmd)
    return commands


def _run_commands(commands: list[list[str]], workdir: Path) -> None:
    for index, cmd in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {' '.join(shlex.quote(part) for part in cmd)}")
        subprocess.run(cmd, cwd=str(workdir), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the online backbone training suite and collect the summary."
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable on AutoDL.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root on AutoDL.",
    )
    parser.add_argument("--cache-dir", required=True, help="Uploaded online cache dir.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory that will store trained model folders.",
    )
    parser.add_argument(
        "--summary-dir",
        required=True,
        help="Directory that will store compare_online_backbones outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_OUTPUTS.keys()),
        choices=list(DEFAULT_MODEL_OUTPUTS.keys()),
        help="Subset of online backbones to run.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--mode",
        default="",
        help="Optional protocol tag forwarded to train_online_detector.py.",
    )
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=2,
        help="Layer count override for transformer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not execute.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_root = Path(args.output_root).resolve()
    summary_dir = Path(args.summary_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    commands = build_commands(
        python_bin=args.python_bin,
        repo_root=repo_root,
        cache_dir=cache_dir,
        output_root=output_root,
        summary_dir=summary_dir,
        models=list(args.models),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        mode=args.mode,
        num_layers_by_model={"transformer": int(args.transformer_layers)},
    )

    if args.dry_run:
        for cmd in commands:
            print(" ".join(shlex.quote(part) for part in cmd))
        return 0

    _run_commands(commands, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
