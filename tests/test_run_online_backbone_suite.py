from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
from scripts.eval.run_online_backbone_suite import _subprocess_env, build_commands


def test_build_commands_covers_all_online_backbones_and_summary(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache_online_rfly_v2"
    output_root = tmp_path / "models_out"
    summary_dir = tmp_path / "summary"

    commands = build_commands(
        python_bin="/usr/bin/python3",
        repo_root=PROJECT_ROOT,
        cache_dir=cache_dir,
        output_root=output_root,
        summary_dir=summary_dir,
        models=["cnn1d", "bilstm", "transformer", "mstca", "kg_mstca"],
        epochs=80,
        batch_size=512,
        lr=3e-4,
        hidden_size=64,
        dropout=0.2,
        mode="fewshot",
        num_layers_by_model={"transformer": 2},
    )

    assert len(commands) == 6

    train_commands = commands[:-1]
    compare_command = commands[-1]

    assert any("--model" in cmd and "kg_mstca" in cmd for cmd in train_commands)
    assert any("online_detector_kg_mstca_rfly" in cmd for cmd in train_commands for cmd in cmd)
    assert any("online_detector_cnn1d_rfly" in cmd for cmd in train_commands for cmd in cmd)

    for cmd in train_commands:
        assert str(cache_dir) in cmd
        assert "--epochs" in cmd
        assert "80" in cmd
        assert "--batch-size" in cmd
        assert "512" in cmd
        assert "--mode" in cmd
        assert "fewshot" in cmd

    assert str(PROJECT_ROOT / "scripts" / "eval" / "compare_online_backbones.py") in compare_command
    assert str(output_root) in compare_command
    assert str(summary_dir) in compare_command
    assert any("kg_mstca=online_detector_kg_mstca_rfly" == item for item in compare_command)


def test_subprocess_env_adds_local_packages(tmp_path: Path) -> None:
    env = _subprocess_env(tmp_path, {"PYTHONPATH": "/existing"})
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[:3] == [
        str(tmp_path),
        str(tmp_path / "packages" / "uav_fault"),
        str(tmp_path / "packages" / "rfly_bridge"),
    ]
    assert parts[-1] == "/existing"
