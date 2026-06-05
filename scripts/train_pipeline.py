#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONLINE_MODELS = ("kg_mstca", "mstca", "cnn1d")
ALL_STAGES = ("splits", "cache", "train", "summary")


@dataclass(frozen=True)
class PipelineCommand:
    label: str
    argv: list[str]


def _is_windows_drive_path(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[2] in ("\\", "/")


def _coerce_cli_path(value: str | Path) -> Path:
    raw = str(value)
    if os.name != "nt" and _is_windows_drive_path(raw):
        drive = raw[0].lower()
        tail = raw[3:].replace("\\", "/")
        return Path("/mnt") / drive / tail
    return Path(raw)


def _default_data_root() -> Path:
    if os.name == "nt":
        return Path(r"D:\UAV-Data")
    candidate = Path("/mnt/d/UAV-Data")
    if candidate.exists():
        return candidate
    return Path("D:/UAV-Data")


def _default_run_name() -> str:
    return "pipeline_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _path_arg(path: Path) -> str:
    return str(path)


def _script(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def _join_for_display(argv: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _extend_pythonpath(env: dict[str, str], repo_root: Path) -> None:
    additions = [
        str(repo_root),
        str(repo_root / "packages" / "uav_fault"),
        str(repo_root / "packages" / "rfly_bridge"),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([*additions, existing]) if existing else os.pathsep.join(additions)


def _extract_tail_after_uav_data(path_text: str) -> str | None:
    normalized = path_text.replace("\\", "/")
    lowered = normalized.lower()
    marker = "/uav-data/"
    idx = lowered.find(marker)
    if idx >= 0:
        return normalized[idx + len(marker) :]
    drive_marker = ":/uav-data/"
    idx = lowered.find(drive_marker)
    if idx >= 0:
        return normalized[idx + len(drive_marker) :]
    if lowered.startswith("uav-data/"):
        return normalized[len("uav-data/") :]
    return None


def _localize_source_path(path_text: str, data_root: Path) -> str:
    tail = _extract_tail_after_uav_data(path_text)
    if tail:
        return str(data_root / Path(tail))
    if os.name != "nt" and _is_windows_drive_path(path_text):
        return str(_coerce_cli_path(path_text))
    return path_text


def prepare_local_manifest(source_manifest: Path, output_manifest: Path, data_root: Path) -> dict[str, int]:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    total = 0
    with source_manifest.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"manifest has no header: {source_manifest}")
        fieldnames = list(reader.fieldnames)
        if "source_csv_path" not in fieldnames:
            raise ValueError(f"manifest missing source_csv_path: {source_manifest}")
        rows = []
        for row in reader:
            total += 1
            before = row.get("source_csv_path", "")
            after = _localize_source_path(before, data_root)
            if after != before:
                converted += 1
            row["source_csv_path"] = after
            rows.append(row)

    with output_manifest.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": total, "converted_source_paths": converted}


def _offline_commands(args: argparse.Namespace, manifest_path: Path) -> list[PipelineCommand]:
    offline_root = args.work_root / "offline_transfer3"
    split_dir = offline_root / "splits_transfer3_v1"
    cache_dir = offline_root / "cache_transfer3_fewshot_v1"
    results_root = offline_root / "results"
    model_dir = results_root / "transfer3_cnnlstm_normalproto"
    summary_dir = offline_root / "summary"

    commands: list[PipelineCommand] = []
    if "splits" in args.stages:
        commands.append(
            PipelineCommand(
                "offline/splits transfer3",
                [
                    args.python_bin,
                    _script("packages", "uav_fault", "uav_fault", "data", "build_transfer3_splits.py"),
                    "--manifest",
                    _path_arg(manifest_path),
                    "--output-dir",
                    _path_arg(split_dir),
                    "--seed",
                    str(args.seed),
                ],
            )
        )
    if "cache" in args.stages:
        commands.append(
            PipelineCommand(
                "offline/cache 3class fewshot",
                [
                    args.python_bin,
                    _script("packages", "uav_fault", "uav_fault", "data", "build_sequence_cache_3class.py"),
                    "--split-dir",
                    _path_arg(split_dir),
                    "--output",
                    _path_arg(cache_dir),
                    "--train-splits",
                    "train",
                    "real_adapt_train",
                    "--label-column",
                    "label_main_3class",
                    "--seq-len",
                    str(args.offline_seq_len),
                ],
            )
        )
    if "train" in args.stages:
        commands.append(
            PipelineCommand(
                "offline/train cnn_lstm_v2_normalproto",
                [
                    args.python_bin,
                    _script("packages", "uav_fault", "uav_fault", "models", "train_transfer3_normalproto.py"),
                    "--mode",
                    "fewshot",
                    "--model",
                    "cnn_lstm_v2_normalproto",
                    "--cache-dir",
                    _path_arg(cache_dir),
                    "--output-dir",
                    _path_arg(model_dir),
                    "--epochs",
                    str(args.offline_epochs),
                    "--batch-size",
                    str(args.offline_batch_size),
                    "--lr",
                    str(args.offline_lr),
                    "--num-workers",
                    str(args.num_workers),
                    "--seed",
                    str(args.seed),
                ],
            )
        )
    if "summary" in args.stages:
        commands.append(
            PipelineCommand(
                "offline/summary",
                [
                    args.python_bin,
                    _script("scripts", "eval", "compare_transfer3_backbones.py"),
                    "--results-root",
                    _path_arg(results_root),
                    "--output-dir",
                    _path_arg(summary_dir),
                    "--experiment",
                    "cnnlstm_normalproto=transfer3_cnnlstm_normalproto",
                ],
            )
        )
    return commands


def _online_commands(args: argparse.Namespace, manifest_path: Path) -> list[PipelineCommand]:
    online_root = args.work_root / "online_real3_binary"
    online_split_dir = online_root / "splits_online_real3_binary_v1"
    cache_dir = online_root / "cache_online_real3_binary_v1"
    results_root = online_root / "models"
    summary_dir = online_root / "summary"

    commands: list[PipelineCommand] = []
    if "splits" in args.stages:
        commands.append(
            PipelineCommand(
                "online/splits real3 binary",
                [
                    args.python_bin,
                    _script("packages", "uav_fault", "uav_fault", "data", "build_online_real3_splits.py"),
                    "--manifest",
                    _path_arg(manifest_path),
                    "--output-dir",
                    _path_arg(online_split_dir),
                    "--seed",
                    str(args.seed),
                ],
            )
        )
    if "cache" in args.stages:
        commands.append(
            PipelineCommand(
                "online/cache binary fewshot windows",
                [
                    args.python_bin,
                    _script("packages", "uav_fault", "uav_fault", "data", "build_online_cache.py"),
                    "--csv-dir",
                    _path_arg(args.data_root),
                    "--splits",
                    _path_arg(online_split_dir / "splits_online_real3_binary_v1.json"),
                    "--output",
                    _path_arg(cache_dir),
                    "--manifest-path",
                    _path_arg(manifest_path),
                    "--window-size",
                    str(args.online_window_size),
                    "--stride",
                    str(args.online_stride),
                    "--target-hz",
                    str(args.target_hz),
                    "--fault-threshold",
                    str(args.fault_threshold),
                    "--max-windows-per-file",
                    str(args.max_windows_per_file),
                    "--protocol-name",
                    "rflymad_online_real3_binary_v1",
                    "--label-space",
                    "real3",
                    "--source-fault-classes",
                    "motor",
                    "sensor",
                ],
            )
        )
    if "train" in args.stages:
        commands.append(
            PipelineCommand(
                "online/train backbone suite",
                [
                    args.python_bin,
                    _script("scripts", "eval", "run_online_backbone_suite.py"),
                    "--python-bin",
                    args.python_bin,
                    "--repo-root",
                    _path_arg(REPO_ROOT),
                    "--cache-dir",
                    _path_arg(cache_dir),
                    "--output-root",
                    _path_arg(results_root),
                    "--summary-dir",
                    _path_arg(summary_dir),
                    "--models",
                    *args.online_models,
                    "--epochs",
                    str(args.online_epochs),
                    "--batch-size",
                    str(args.online_batch_size),
                    "--lr",
                    str(args.online_lr),
                    "--hidden-size",
                    str(args.online_hidden_size),
                    "--dropout",
                    str(args.online_dropout),
                    "--mode",
                    "fewshot",
                ],
            )
        )
    elif "summary" in args.stages:
        commands.append(
            PipelineCommand(
                "online/summary",
                [
                    args.python_bin,
                    _script("scripts", "eval", "compare_online_backbones.py"),
                    "--results-root",
                    _path_arg(results_root),
                    "--output-dir",
                    _path_arg(summary_dir),
                    *[
                        item
                        for model in args.online_models
                        for item in (
                            "--experiment",
                            f"{model}=online_detector_{model}_rfly",
                        )
                    ],
                ],
            )
        )
    return commands


def build_commands(args: argparse.Namespace, manifest_path: Path) -> list[PipelineCommand]:
    commands: list[PipelineCommand] = []
    if args.task in ("offline", "both"):
        commands.extend(_offline_commands(args, manifest_path))
    if args.task in ("online", "both"):
        commands.extend(_online_commands(args, manifest_path))
    return commands


def _validate_paths(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(f"data root not found: {args.data_root}")
    if not args.source_manifest.exists():
        raise FileNotFoundError(f"manifest not found: {args.source_manifest}")


def _print_plan(args: argparse.Namespace, manifest_path: Path, commands: list[PipelineCommand]) -> None:
    print("UAV training pipeline")
    print(f"  task          : {args.task}")
    print(f"  mode          : {'execute' if args.execute else 'dry-run'}")
    print(f"  repo_root     : {REPO_ROOT}")
    print(f"  data_root     : {args.data_root}")
    print(f"  work_root     : {args.work_root}")
    print(f"  manifest      : {manifest_path}")
    print(f"  stages        : {', '.join(args.stages)}")
    print(f"  python_bin    : {args.python_bin}")
    print("  PYTHONPATH    : repo root + packages/uav_fault + packages/rfly_bridge")
    print()
    if not commands:
        print("No commands selected.")
        return
    for idx, command in enumerate(commands, start=1):
        print(f"[{idx}/{len(commands)}] {command.label}")
        print("  " + _join_for_display(command.argv))


def _run_commands(commands: list[PipelineCommand], repo_root: Path) -> None:
    env = os.environ.copy()
    _extend_pythonpath(env, repo_root)
    for idx, command in enumerate(commands, start=1):
        print(f"\n[{idx}/{len(commands)}] {command.label}")
        print(_join_for_display(command.argv))
        subprocess.run(command.argv, cwd=str(repo_root), env=env, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible RflyMAD transfer3/offline and online training pipelines."
    )
    parser.add_argument("--task", choices=("offline", "online", "both"), default="both")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=list(ALL_STAGES),
        help="Pipeline stages to include. Default: all.",
    )
    parser.add_argument("--data-root", default=str(_default_data_root()))
    parser.add_argument("--manifest", default="")
    parser.add_argument("--work-root", default="")
    parser.add_argument("--run-name", default=_default_run_name())
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--execute", action="store_true", help="Run commands. Default is dry-run.")
    parser.add_argument("--force", action="store_true", help="Allow a non-empty work root.")
    parser.add_argument(
        "--skip-local-manifest",
        action="store_true",
        help="Use the source manifest directly instead of writing a path-local copy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--offline-seq-len", type=int, default=2048)
    parser.add_argument("--offline-epochs", type=int, default=120)
    parser.add_argument("--offline-batch-size", type=int, default=32)
    parser.add_argument("--offline-lr", type=float, default=3e-4)

    parser.add_argument("--online-models", nargs="+", default=list(DEFAULT_ONLINE_MODELS))
    parser.add_argument("--online-window-size", type=int, default=150)
    parser.add_argument("--online-stride", type=int, default=10)
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--fault-threshold", type=float, default=0.5)
    parser.add_argument("--max-windows-per-file", type=int, default=8)
    parser.add_argument("--online-epochs", type=int, default=100)
    parser.add_argument("--online-batch-size", type=int, default=512)
    parser.add_argument("--online-lr", type=float, default=3e-4)
    parser.add_argument("--online-hidden-size", type=int, default=64)
    parser.add_argument("--online-dropout", type=float, default=0.2)

    args = parser.parse_args(argv)
    args.data_root = _coerce_cli_path(args.data_root)
    args.source_manifest = (
        _coerce_cli_path(args.manifest)
        if args.manifest
        else args.data_root / "manifest" / "manifest_v2.csv"
    )
    args.work_root = (
        _coerce_cli_path(args.work_root)
        if args.work_root
        else args.data_root / "training_runs" / args.run_name
    )
    args.stages = tuple(dict.fromkeys(args.stages))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_paths(args)

    local_manifest = args.work_root / "manifest" / "manifest_v2.local.csv"
    manifest_for_commands = args.source_manifest if args.skip_local_manifest else local_manifest
    commands = build_commands(args, manifest_for_commands)
    _print_plan(args, manifest_for_commands, commands)

    if not args.execute:
        print("\nDry-run only. Add --execute to run the pipeline.")
        return 0

    if args.work_root.exists() and any(args.work_root.iterdir()) and not args.force:
        raise SystemExit(
            f"work root is not empty: {args.work_root}\n"
            "Use a new --run-name/--work-root or pass --force."
        )
    args.work_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_local_manifest:
        summary = prepare_local_manifest(args.source_manifest, local_manifest, args.data_root)
        print(
            f"\nprepared local manifest: {local_manifest} "
            f"({summary['converted_source_paths']}/{summary['rows']} source paths converted)"
        )

    _run_commands(commands, REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
