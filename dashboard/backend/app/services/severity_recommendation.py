from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import current_app

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class RuleMetric:
    key: str
    direction: str
    medium: float
    heavy: float
    weight: float
    required_for_decision: bool


_RULE_CACHE: dict[str, Any] = {
    'path': None,
    'mtime': None,
    'rules': None,
}


_DEFAULT_RULES = {
    'version': 'rule_v1',
    'severity_levels': {'light_max': 0.45, 'medium_max': 0.72},
    'defaults': {
        'confidence_metric': {
            'key': 'prediction_confidence',
            'direction': 'high',
            'medium': 0.6,
            'heavy': 0.85,
            'weight': 0.1,
            'required_for_decision': False,
        }
    },
    'faults': {},
}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if math.isnan(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def _numeric_stat(df: pd.DataFrame, column: str, stat: str) -> float | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors='coerce').dropna()
    if values.empty:
        return None
    if stat == 'min':
        return float(values.min())
    if stat == 'max':
        return float(values.max())
    if stat == 'mean':
        return float(values.mean())
    if stat == 'p95':
        return float(values.quantile(0.95))
    return None


def _load_rules(config_path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(config_path) if config_path else Path(current_app.config['SEVERITY_RULES_PATH'])
    cache_key = str(cfg_path)
    mtime = cfg_path.stat().st_mtime if cfg_path.exists() else None

    if _RULE_CACHE['path'] == cache_key and _RULE_CACHE['mtime'] == mtime and _RULE_CACHE['rules'] is not None:
        return _RULE_CACHE['rules']

    rules = _DEFAULT_RULES
    if cfg_path.exists():
        text = cfg_path.read_text(encoding='utf-8')
        loaded = None
        if yaml is not None:
            try:
                loaded = yaml.safe_load(text)
            except Exception:
                loaded = None
        if loaded is None:
            try:
                loaded = json.loads(text)
            except Exception:
                loaded = None
        if isinstance(loaded, dict):
            rules = loaded
        else:
            current_app.logger.warning('强度规则配置解析失败，回退默认规则: %s', cfg_path)
    else:
        current_app.logger.warning('未找到强度规则配置，回退默认规则: %s', cfg_path)

    _RULE_CACHE['path'] = cache_key
    _RULE_CACHE['mtime'] = mtime
    _RULE_CACHE['rules'] = rules
    return rules


def _load_csv(csv_path: str) -> pd.DataFrame:
    target_cols = {
        'battery_warning', 'battery_faults', 'battery_remaining', 'battery_voltage_v',
        'motor1', 'motor2', 'motor3', 'motor4',
        'esc_rpm_1', 'esc_rpm_2', 'esc_rpm_3', 'esc_rpm_4',
        'gyro_vibration_metric', 'accel_vibration_metric', 'imbalanced_prop_metric',
        'thrust_z', 'unalloc_thrust_z', 'vz', 'fault_intensity', 'fault_severity',
    }
    try:
        return pd.read_csv(csv_path, usecols=lambda c: c in target_cols)
    except Exception:
        return pd.read_csv(csv_path)


def _extract_metrics(df: pd.DataFrame, confidence: float | None) -> dict[str, float]:
    metrics: dict[str, float] = {}

    confidence_value = _as_float(confidence)
    if confidence_value is not None:
        metrics['prediction_confidence'] = confidence_value

    for col, stat, key in (
        ('battery_warning', 'max', 'battery_warning_max'),
        ('battery_faults', 'max', 'battery_faults_max'),
        ('battery_remaining', 'min', 'battery_remaining_min'),
        ('fault_intensity', 'mean', 'fault_intensity_mean'),
        ('fault_severity', 'p95', 'fault_severity_p95'),
        ('fault_severity', 'max', 'fault_severity_max'),
        ('gyro_vibration_metric', 'p95', 'gyro_vibration_p95'),
        ('accel_vibration_metric', 'p95', 'accel_vibration_p95'),
        ('imbalanced_prop_metric', 'p95', 'imbalanced_prop_metric_p95'),
        ('unalloc_thrust_z', 'p95', 'unalloc_thrust_z_p95'),
    ):
        value = _numeric_stat(df, col, stat)
        if value is not None:
            metrics[key] = value

    voltage_max = _numeric_stat(df, 'battery_voltage_v', 'max')
    voltage_min = _numeric_stat(df, 'battery_voltage_v', 'min')
    if voltage_max is not None and voltage_min is not None and voltage_max > 1e-6:
        metrics['battery_voltage_drop_ratio'] = max(0.0, (voltage_max - voltage_min) / voltage_max)

    motor_cols = ['motor1', 'motor2', 'motor3', 'motor4']
    if all(col in df.columns for col in motor_cols):
        motors = df[motor_cols].apply(pd.to_numeric, errors='coerce')
        row_mean = motors.mean(axis=1).replace(0, np.nan)
        row_std = motors.std(axis=1)
        ratio = (row_std / row_mean.abs()).replace([np.inf, -np.inf], np.nan).dropna()
        if not ratio.empty:
            metrics['motor_imbalance_ratio'] = float(ratio.quantile(0.95))

    esc_cols = ['esc_rpm_1', 'esc_rpm_2', 'esc_rpm_3', 'esc_rpm_4']
    if all(col in df.columns for col in esc_cols):
        esc = df[esc_cols].apply(pd.to_numeric, errors='coerce')
        row_mean = esc.mean(axis=1).replace(0, np.nan)
        row_std = esc.std(axis=1)
        ratio = (row_std / row_mean.abs()).replace([np.inf, -np.inf], np.nan).dropna()
        if not ratio.empty:
            metrics['esc_rpm_imbalance_ratio'] = float(ratio.quantile(0.95))

    if 'thrust_z' in df.columns:
        thrust = pd.to_numeric(df['thrust_z'], errors='coerce').dropna()
        if not thrust.empty:
            metrics['thrust_saturation_ratio'] = float((thrust.abs() >= 0.85).mean())

    if 'vz' in df.columns:
        vz = pd.to_numeric(df['vz'], errors='coerce').dropna()
        if not vz.empty:
            metrics['vz_down_p95'] = float((-vz).clip(lower=0).quantile(0.95))

    return metrics


def _metric_from_config(item: dict[str, Any]) -> RuleMetric:
    return RuleMetric(
        key=str(item.get('key', '')).strip(),
        direction=str(item.get('direction', 'high')).strip().lower(),
        medium=float(item.get('medium', 0.0)),
        heavy=float(item.get('heavy', 1.0)),
        weight=float(item.get('weight', 0.0)),
        required_for_decision=bool(item.get('required_for_decision', True)),
    )


def _score_metric(value: float, metric: RuleMetric) -> float:
    if metric.direction not in ('high', 'low'):
        return 0.0

    if metric.direction == 'high':
        medium = metric.medium
        heavy = metric.heavy if metric.heavy > metric.medium else metric.medium + 1e-6
        if value <= medium:
            return 0.0
        if value >= heavy:
            return 1.0
        return 0.5 + 0.49 * (value - medium) / (heavy - medium)

    medium = metric.medium
    heavy = metric.heavy if metric.heavy < metric.medium else metric.medium - 1e-6
    if value >= medium:
        return 0.0
    if value <= heavy:
        return 1.0
    return 0.5 + 0.49 * (medium - value) / (medium - heavy)


def _score_to_level(score: float, levels_cfg: dict[str, Any]) -> str:
    light_max = float(levels_cfg.get('light_max', 0.45))
    medium_max = float(levels_cfg.get('medium_max', 0.72))
    if score < light_max:
        return 'light'
    if score < medium_max:
        return 'medium'
    return 'heavy'


def _normalize_recommendations(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, str):
            output.append({'title': item, 'priority': 'P2', 'reason_codes': ['legacy_text']})
            continue
        if not isinstance(item, dict):
            continue
        title = str(item.get('title', '')).strip()
        if not title:
            continue
        reason_codes = item.get('reason_codes')
        if not isinstance(reason_codes, list):
            reason_codes = []
        output.append({
            'title': title,
            'priority': str(item.get('priority', 'P2')).strip() or 'P2',
            'reason_codes': [str(code) for code in reason_codes if str(code).strip()],
        })
    return output


def _choose_recommendations(rec_map: dict[str, Any], severity_level: str) -> tuple[list[dict[str, Any]], bool]:
    fallback_used = False
    if severity_level == 'unknown':
        recs = _normalize_recommendations(rec_map.get('fallback'))
        if not recs:
            recs = _normalize_recommendations(rec_map.get('medium'))
        fallback_used = True
        return recs, fallback_used

    recs = _normalize_recommendations(rec_map.get(severity_level))
    if not recs:
        recs = _normalize_recommendations(rec_map.get('fallback'))
        fallback_used = True
    return recs, fallback_used


def _level_by_primary_metric(value: float, cfg: dict[str, Any]) -> str:
    direction = str(cfg.get('direction', 'high')).strip().lower()
    light_max = float(cfg.get('light_max', 0.2))
    medium_max = float(cfg.get('medium_max', 0.5))

    if direction == 'low':
        if value >= light_max:
            return 'light'
        if value >= medium_max:
            return 'medium'
        return 'heavy'

    if value <= light_max:
        return 'light'
    if value <= medium_max:
        return 'medium'
    return 'heavy'


def evaluate_severity_and_recommendations(
    csv_path: str,
    fault_label: str,
    confidence: float | None,
    config_path: str | None = None,
) -> dict[str, Any]:
    rules = _load_rules(config_path=config_path)
    strategy = str(rules.get('version', 'rule_v1'))
    fault_cfg = (rules.get('faults') or {}).get(fault_label, {})

    if str(fault_cfg.get('mode', '')).lower() == 'fixed':
        severity = {
            'level': str(fault_cfg.get('level', 'none')),
            'score': 0.0,
            'method': strategy,
            'evidence': [],
        }
        recs = _normalize_recommendations(fault_cfg.get('recommendations'))
        return {
            'severity': severity,
            'recommendations': recs,
            'recommendation_meta': {
                'strategy': strategy,
                'fallback_used': False,
                'severity_level': severity['level'],
                'candidate_count': len(recs),
            },
        }

    df = _load_csv(csv_path)
    metrics = _extract_metrics(df, confidence=confidence)
    rec_map = fault_cfg.get('recommendations') or {}

    primary_metric_cfg = fault_cfg.get('primary_metric')
    if isinstance(primary_metric_cfg, dict):
        metric_key = str(primary_metric_cfg.get('key', '')).strip()
        metric_value = metrics.get(metric_key)
        if metric_key and metric_value is not None:
            severity_level = _level_by_primary_metric(float(metric_value), primary_metric_cfg)
            recs, fallback_used = _choose_recommendations(rec_map, severity_level)
            severity_score = _as_float(metric_value)
            severity = {
                'level': severity_level,
                'score': None if severity_score is None else round(float(severity_score), 4),
                'method': f'{strategy}_primary',
                'evidence': [{
                    'metric': metric_key,
                    'status': 'used',
                    'value': float(metric_value),
                    'rule': {
                        'direction': str(primary_metric_cfg.get('direction', 'high')),
                        'light_max': float(primary_metric_cfg.get('light_max', 0.2)),
                        'medium_max': float(primary_metric_cfg.get('medium_max', 0.5)),
                    },
                }],
            }
            return {
                'severity': severity,
                'recommendations': recs,
                'recommendation_meta': {
                    'strategy': strategy,
                    'fallback_used': fallback_used,
                    'severity_level': severity_level,
                    'candidate_count': len(recs),
                    'mode': 'primary_metric',
                },
            }

    metric_items = list(fault_cfg.get('metrics') or [])
    confidence_metric = (rules.get('defaults') or {}).get('confidence_metric')
    if isinstance(confidence_metric, dict):
        metric_items.append(confidence_metric)

    metric_rules = [_metric_from_config(item) for item in metric_items if item.get('key')]
    evidence: list[dict[str, Any]] = []
    weighted_sum = 0.0
    weight_sum = 0.0
    required_available = 0

    for metric in metric_rules:
        raw_value = metrics.get(metric.key)
        if raw_value is None:
            evidence.append({
                'metric': metric.key,
                'status': 'missing',
                'required': metric.required_for_decision,
                'reason': 'missing_or_non_numeric',
            })
            continue

        score = _score_metric(raw_value, metric)
        weight = max(0.0, metric.weight)
        weighted_sum += score * weight
        weight_sum += weight
        if metric.required_for_decision:
            required_available += 1
        evidence.append({
            'metric': metric.key,
            'status': 'used',
            'value': float(raw_value),
            'score': round(float(score), 4),
            'weight': weight,
            'required': metric.required_for_decision,
        })

    min_required = int(fault_cfg.get('min_required_metrics', 1))
    fallback_used = False
    if required_available < min_required or weight_sum <= 1e-9:
        severity_level = 'unknown'
        severity_score = None
        fallback_used = True
    else:
        severity_score = float(weighted_sum / weight_sum)
        severity_level = _score_to_level(severity_score, rules.get('severity_levels') or {})

    recs, rec_fallback = _choose_recommendations(rec_map, severity_level)
    fallback_used = bool(fallback_used or rec_fallback)

    severity = {
        'level': severity_level,
        'score': None if severity_score is None else round(severity_score, 4),
        'method': strategy,
        'evidence': evidence,
    }
    return {
        'severity': severity,
        'recommendations': recs,
        'recommendation_meta': {
            'strategy': strategy,
            'fallback_used': fallback_used,
            'severity_level': severity_level,
            'candidate_count': len(recs),
            'required_metrics': {
                'need': min_required,
                'available': required_available,
            },
        },
    }
