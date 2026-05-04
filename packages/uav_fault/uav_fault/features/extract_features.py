"""
从每个 CSV 文件提取统计特征，生成表格化数据集。
输出: data/processed/features.csv
"""
import os
import csv
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

# ── 特征列定义 ──

FEATURE_COLUMNS = [
    'motor1', 'motor2', 'motor3', 'motor4',
    'roll', 'pitch', 'yaw',
    'rollspeed', 'pitchspeed', 'yawspeed',
    'pos_z', 'alt_local_m',
    'vx', 'vy', 'vz', 'speed_xy_m_s', 'speed_3d_m_s',
    'accel_x', 'accel_y', 'accel_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'mag_x', 'mag_y', 'mag_z',
    'baro_alt_meter', 'baro_pressure_pa',
    'battery_voltage_v', 'battery_remaining',
    'dist_bottom',
    # 控制器内部状态
    'hover_thrust', 'accel_innov', 'accel_innov_test_ratio',
    'rollspeed_integ', 'pitchspeed_integ', 'yawspeed_integ',
    'thrust_z',
    'torque_roll', 'torque_pitch', 'torque_yaw',
    # 振动指标
    'accel_vibration_metric', 'gyro_vibration_metric',
    # 角速率设定值与跟踪误差
    'rates_sp_roll', 'rates_sp_pitch', 'rates_sp_yaw',
    'rate_err_roll', 'rate_err_pitch', 'rate_err_yaw',
    # ESC 转速
    'esc_rpm_1', 'esc_rpm_2', 'esc_rpm_3', 'esc_rpm_4',
]

LABEL_MAP = {
    'nf': 'normal',
    'bf': 'battery_failure',
    'mf': 'motor_failure',
    'pi': 'prop_imbalance',
    'tf': 'thrust_loss',
}

# 标签泄漏列：这些列直接或间接编码了故障信息，必须在特征提取前丢弃
DROP_COLS = [
    'is_fault_active', 'fault_intensity', 'time_to_fault', 'fault_severity',
    'flight_phase',  # 仅用于分段，不应作为特征
]

# Phase-aware 特征提取用的列和统计量
PHASE_AWARE_COLUMNS = [
    'motor1', 'motor2', 'motor3', 'motor4',
    'roll', 'pitch', 'accel_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'vz', 'speed_xy_m_s',
    'acc_norm', 'motor_cross_mean', 'motor_cross_std',
]

PHASE_STATS = ['mean', 'std', 'min', 'max', 'p10', 'p90']


# ── 统计量计算 ──

def compute_stats(series: np.ndarray) -> dict:
    """对单个通道的时序数据计算统计特征。"""
    s = series[np.isfinite(series)]
    if len(s) == 0:
        return {k: np.nan for k in [
            'mean', 'std', 'min', 'max', 'median',
            'p10', 'p90', 'iqr', 'rms', 'skew', 'kurtosis',
            'first_half_mean', 'first_half_std',
            'second_half_mean', 'second_half_std', 'delta_mean',
        ]}

    mid = len(s) // 2
    first_half = s[:mid]
    second_half = s[mid:]

    return {
        'mean': np.mean(s),
        'std': np.std(s),
        'min': np.min(s),
        'max': np.max(s),
        'median': np.median(s),
        'p10': np.percentile(s, 10),
        'p90': np.percentile(s, 90),
        'iqr': np.percentile(s, 75) - np.percentile(s, 25),
        'rms': np.sqrt(np.mean(s ** 2)),
        'skew': float(sp_stats.skew(s)),
        'kurtosis': float(sp_stats.kurtosis(s)),
        'first_half_mean': np.mean(first_half),
        'first_half_std': np.std(first_half),
        'second_half_mean': np.mean(second_half),
        'second_half_std': np.std(second_half),
        'delta_mean': np.mean(second_half) - np.mean(first_half),
    }


def compute_derived_channels(df: pd.DataFrame) -> dict:
    """计算派生通道并返回统计特征。"""
    derived = {}

    # 电机统计
    motors = df[['motor1', 'motor2', 'motor3', 'motor4']].values
    derived['motor_cross_mean'] = np.mean(motors, axis=1)
    derived['motor_cross_std'] = np.std(motors, axis=1)
    derived['motor_cross_range'] = np.max(motors, axis=1) - np.min(motors, axis=1)

    # 加速度模
    derived['acc_norm'] = np.sqrt(
        df['accel_x'].values ** 2 +
        df['accel_y'].values ** 2 +
        df['accel_z'].values ** 2
    )

    # 角速度模
    derived['gyro_norm'] = np.sqrt(
        df['gyro_x'].values ** 2 +
        df['gyro_y'].values ** 2 +
        df['gyro_z'].values ** 2
    )

    # 姿态模
    derived['attitude_norm'] = np.sqrt(
        df['roll'].values ** 2 +
        df['pitch'].values ** 2
    )

    # 控制器努力度: 电机均值 / |accel_z|，捕捉推力损失时控制器补偿的信号
    az_abs = np.abs(df['accel_z'].values)
    derived['motor_effort'] = derived['motor_cross_mean'] / np.clip(az_abs, 0.1, None)

    # ESC RPM 不对称性
    esc_cols = ['esc_rpm_1', 'esc_rpm_2', 'esc_rpm_3', 'esc_rpm_4']
    if all(c in df.columns for c in esc_cols):
        esc_vals = df[esc_cols].values
        derived['esc_rpm_cross_std'] = np.std(esc_vals, axis=1)
        derived['esc_rpm_cross_range'] = np.max(esc_vals, axis=1) - np.min(esc_vals, axis=1)

    features = {}
    for name, values in derived.items():
        for stat_name, stat_val in compute_stats(values).items():
            features[f'{name}__{stat_name}'] = stat_val

    return features


def compute_phase_stats(series: np.ndarray) -> dict:
    """精简版统计量，用于 phase-aware 特征。"""
    s = series[np.isfinite(series)]
    if len(s) == 0:
        return {k: np.nan for k in PHASE_STATS}
    return {
        'mean': np.mean(s),
        'std': np.std(s),
        'min': np.min(s),
        'max': np.max(s),
        'p10': np.percentile(s, 10),
        'p90': np.percentile(s, 90),
    }


def compute_phase_aware_features(df: pd.DataFrame, phase_col: pd.Series = None) -> dict:
    """分别对 normal 段和 fault 段提取特征，计算差值和比值。"""
    features = {}

    if phase_col is not None:
        normal_df = df[phase_col == 'normal']
        fault_df = df[phase_col == 'fault']
    else:
        normal_df = df
        fault_df = pd.DataFrame()

    has_fault = len(fault_df) > 0

    # 准备派生列（如果不在 df 中）
    def _add_derived_cols(sub_df):
        if len(sub_df) == 0 or 'acc_norm' in sub_df.columns:
            return sub_df
        sub_df = sub_df.copy()
        motors = sub_df[['motor1', 'motor2', 'motor3', 'motor4']].values
        sub_df['acc_norm'] = np.sqrt(
            sub_df['accel_x'].values ** 2 +
            sub_df['accel_y'].values ** 2 +
            sub_df['accel_z'].values ** 2
        )
        sub_df['motor_cross_mean'] = np.mean(motors, axis=1)
        sub_df['motor_cross_std'] = np.std(motors, axis=1)
        return sub_df

    normal_df = _add_derived_cols(normal_df)
    fault_df = _add_derived_cols(fault_df)

    for col in PHASE_AWARE_COLUMNS:
        # normal phase
        if col in normal_df.columns and len(normal_df) > 0:
            n_stats = compute_phase_stats(normal_df[col].values.astype(float))
        else:
            n_stats = {k: np.nan for k in PHASE_STATS}

        for stat, val in n_stats.items():
            features[f'normal_phase__{col}__{stat}'] = val

        # fault phase
        if has_fault and col in fault_df.columns:
            f_stats = compute_phase_stats(fault_df[col].values.astype(float))
        else:
            f_stats = {k: np.nan for k in PHASE_STATS}

        for stat, val in f_stats.items():
            features[f'fault_phase__{col}__{stat}'] = val

        # delta = fault - normal
        for stat in PHASE_STATS:
            nv = n_stats[stat]
            fv = f_stats[stat]
            features[f'phase_delta__{col}__{stat}'] = fv - nv if (np.isfinite(fv) and np.isfinite(nv)) else np.nan

        # ratio = fault / normal (避免除零)
        for stat in PHASE_STATS:
            nv = n_stats[stat]
            fv = f_stats[stat]
            if np.isfinite(fv) and np.isfinite(nv) and abs(nv) > 1e-9:
                features[f'phase_ratio__{col}__{stat}'] = fv / nv
            else:
                features[f'phase_ratio__{col}__{stat}'] = np.nan

    return features


def compute_thrust_loss_features(df: pd.DataFrame) -> dict:
    """Thrust-loss 专用物理特征。"""
    features = {}
    motors = df[['motor1', 'motor2', 'motor3', 'motor4']].values

    # motor_cap_ratio: 四电机同时在 [0.48, 0.52] 的行占比
    in_cap = np.all((motors >= 0.48) & (motors <= 0.52), axis=1)
    features['motor_cap_ratio'] = np.mean(in_cap)

    # motor_bandwidth_mean: 四电机 p99-p01 均值
    bandwidths = []
    for i in range(4):
        col = motors[:, i]
        col_clean = col[np.isfinite(col)]
        if len(col_clean) > 0:
            bandwidths.append(np.percentile(col_clean, 99) - np.percentile(col_clean, 1))
        else:
            bandwidths.append(np.nan)
    features['motor_bandwidth_mean'] = np.nanmean(bandwidths)

    # thrust_deficit: 9.81 - |accel_z|
    accel_z = df['accel_z'].values.astype(float)
    deficit = 9.81 - np.abs(accel_z)
    deficit_clean = deficit[np.isfinite(deficit)]
    features['thrust_deficit_mean'] = np.mean(deficit_clean) if len(deficit_clean) > 0 else np.nan
    features['thrust_deficit_max'] = np.max(deficit_clean) if len(deficit_clean) > 0 else np.nan

    # motor_accel_mismatch: 电机均值与 |accel_z| 的相关系数
    motor_mean = np.mean(motors, axis=1)
    abs_az = np.abs(accel_z)
    mask = np.isfinite(motor_mean) & np.isfinite(abs_az)
    if np.sum(mask) > 2:
        corr = np.corrcoef(motor_mean[mask], abs_az[mask])[0, 1]
        features['motor_accel_mismatch'] = corr if np.isfinite(corr) else np.nan
    else:
        features['motor_accel_mismatch'] = np.nan

    return features


# ── 单文件特征提取 ──

def extract_features_from_file(csv_path: str) -> dict:
    """从单个 CSV 文件提取所有特征。"""
    df = pd.read_csv(csv_path)

    # 过滤掉 crashed 阶段的数据（坠毁/断连后的无效数据）
    if 'flight_phase' in df.columns:
        df = df[df['flight_phase'].isin(['normal', 'fault'])].reset_index(drop=True)

    # 提取 flight_phase 用于分段（phase-aware 特征需要），然后丢弃标签泄漏列
    phase_col = df['flight_phase'].copy() if 'flight_phase' in df.columns else None
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 替换 inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    features = {}

    # 基础通道统计特征
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            for stat_name in ['mean', 'std', 'min', 'max', 'median',
                              'p10', 'p90', 'iqr', 'rms', 'skew', 'kurtosis',
                              'first_half_mean', 'first_half_std',
                              'second_half_mean', 'second_half_std', 'delta_mean']:
                features[f'{col}__{stat_name}'] = np.nan
            continue

        values = df[col].values.astype(float)
        for stat_name, stat_val in compute_stats(values).items():
            features[f'{col}__{stat_name}'] = stat_val

    # 派生通道
    required_derived = ['motor1', 'motor2', 'motor3', 'motor4',
                        'accel_x', 'accel_y', 'accel_z',
                        'gyro_x', 'gyro_y', 'gyro_z',
                        'roll', 'pitch']
    if all(c in df.columns for c in required_derived):
        derived_features = compute_derived_channels(df)
        features.update(derived_features)

    # NOTE: Phase-aware 特征已移除 — normal 样本无 fault_phase 导致 100% NaN，
    # 经 imputer 填充后成为完美的标签泄漏指示器。
    # features.update(compute_phase_aware_features(df, phase_col))

    # Thrust-loss 物理特征
    features.update(compute_thrust_loss_features(df))

    return features


# ── 主流程 ──

def build_features(csv_dir: str, manifest_path: str, output_path: str):
    csv_dir = Path(csv_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取 manifest
    manifest = pd.read_csv(manifest_path)
    print(f"从 manifest 读取 {len(manifest)} 个样本")

    all_rows = []
    for idx, row in manifest.iterrows():
        fname = row['filename']
        label = row['label']
        csv_path = csv_dir / fname

        if not csv_path.exists():
            print(f"[WARN] 文件不存在，跳过: {fname}")
            continue

        features = extract_features_from_file(str(csv_path))
        features['filename'] = fname
        features['label'] = label
        all_rows.append(features)

        if (idx + 1) % 50 == 0:
            print(f"  已处理 {idx + 1}/{len(manifest)} ...")

    # 输出
    result_df = pd.DataFrame(all_rows)

    # 把 filename 和 label 放到前面
    cols = ['filename', 'label'] + [c for c in result_df.columns if c not in ('filename', 'label')]
    result_df = result_df[cols]

    # 移除零方差特征（训练集中完全无区分度的列）
    feature_cols = [c for c in result_df.columns if c not in ('filename', 'label')]
    zero_var_cols = [c for c in feature_cols if result_df[c].std() == 0 or result_df[c].isna().all()]
    if zero_var_cols:
        result_df.drop(columns=zero_var_cols, inplace=True)
        print(f"\n[CLEANUP] 移除 {len(zero_var_cols)} 个零方差特征: {zero_var_cols[:10]}{'...' if len(zero_var_cols) > 10 else ''}")

    result_df.to_csv(output_path, index=False)
    n_features = len(result_df.columns) - 2
    print(f"\n特征提取完成: {output_path}")
    print(f"样本数: {len(result_df)}, 特征数: {n_features}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='提取统计特征')
    parser.add_argument('--csv-dir', default='data/csv_raw')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--output', default='data/processed/features.csv')
    args = parser.parse_args()
    build_features(args.csv_dir, args.manifest, args.output)
