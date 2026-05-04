import json
import subprocess
from pathlib import Path

from flask import current_app


def _resolve_prepare_script(cfg) -> Path:
    explicit_script = str(cfg.get('ULG_PREPARE_SCRIPT', '') or '').strip()
    if explicit_script:
        script_path = Path(explicit_script)
    else:
        fault_dataset_root = str(cfg.get('FAULT_DATASET_ROOT', '') or '').strip()
        if not fault_dataset_root:
            raise FileNotFoundError(
                '未配置 ULG 转换脚本。请设置 ULG_PREPARE_SCRIPT，'
                '或将 FAULT_DATASET_ROOT 指向 Fault Dataset 根目录。'
            )
        script_path = Path(fault_dataset_root) / 'scripts' / 'tools' / 'build_fault_dataset.py'

    if not script_path.is_absolute():
        script_path = (Path(cfg['REPO_ROOT']) / script_path).resolve()

    if not script_path.exists():
        raise FileNotFoundError(
            f'未找到转换脚本: {script_path}。'
            '请检查 ULG_PREPARE_SCRIPT 或 FAULT_DATASET_ROOT 配置。'
        )
    return script_path


def run_ulg_to_csv(ulg_path: str, output_csv: str) -> dict:
    """调用 Fault Dataset 工具将 ULG 转为 CSV。"""
    cfg = current_app.config
    python_bin = cfg['ML_PYTHON_BIN']
    script_path = _resolve_prepare_script(cfg)

    cmd = [
        python_bin,
        str(script_path),
        '--ulg_path',
        str(ulg_path),
        '--output_csv',
        str(output_csv),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cfg['REPO_ROOT'],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            'ULG转换失败: '
            f'returncode={completed.returncode}\n'
            f'stdout={completed.stdout[-4000:]}\n'
            f'stderr={completed.stderr[-4000:]}'
        )

    return {
        'command': cmd,
        'stdout_tail': completed.stdout[-2000:],
        'stderr_tail': completed.stderr[-2000:],
        'output_csv': output_csv,
    }


def run_offline_5class(csv_path: str, model_dir: str, output_json: str) -> dict:
    """调用离线5分类推理桥接脚本。"""
    cfg = current_app.config
    python_bin = cfg['ML_PYTHON_BIN']
    script_path = Path(__file__).resolve().parents[2] / 'scripts' / 'offline5_infer.py'
    if not script_path.exists():
        raise FileNotFoundError(f'未找到推理脚本: {script_path}')

    cmd = [
        python_bin,
        str(script_path),
        '--repo-root',
        cfg['REPO_ROOT'],
        '--csv-path',
        str(csv_path),
        '--model-dir',
        str(model_dir),
        '--norm-stats',
        cfg.get('OFFLINE_NORM_STATS', ''),
        '--output-json',
        str(output_json),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cfg['REPO_ROOT'],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            '离线检测失败: '
            f'returncode={completed.returncode}\n'
            f'stdout={completed.stdout[-4000:]}\n'
            f'stderr={completed.stderr[-4000:]}'
        )
    result_path = Path(output_json)
    if not result_path.exists():
        raise FileNotFoundError(f'检测结果缺失: {output_json}')

    with result_path.open('r', encoding='utf-8') as fp:
        data = json.load(fp)

    data['stdout_tail'] = completed.stdout[-2000:]
    data['stderr_tail'] = completed.stderr[-2000:]
    return data
