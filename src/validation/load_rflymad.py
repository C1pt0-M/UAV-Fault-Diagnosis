#!/usr/bin/env python3
"""
RflyMAD 数据加载器
将处理后的 RflyMAD CSV 转换为模型输入格式
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation

def quaternion_to_euler(q):
    """
    将四元数转换为欧拉角 (roll, pitch, yaw)
    q: [w, x, y, z] 或 [x, y, z, w] 格式
    返回: (roll, pitch, yaw) in radians
    """
    # RflyMAD 使用 [w, x, y, z] 格式
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy 需要 [x,y,z,w]
    return r.as_euler('xyz', degrees=False)

def load_rflymad_case(csv_path):
    """
    加载单个 RflyMAD 案例

    返回:
        data: (T, 12) numpy array - 12 通道数据
        label: int - 故障标签 (0=normal, 1=fault)
        metadata: dict - 元数据
    """
    df = pd.read_csv(csv_path)

    # 检测数据类型 (SIL/HIL vs Real)
    is_simulation = 'trueTime' in df.columns

    # 提取 12 个核心通道
    # 1. 四元数 -> 欧拉角
    q_cols = ['_estimator_attitude_0_q[0]', '_estimator_attitude_0_q[1]',
              '_estimator_attitude_0_q[2]', '_estimator_attitude_0_q[3]']

    euler_angles = []
    for idx, row in df.iterrows():
        q = [row[col] for col in q_cols]
        roll, pitch, yaw = quaternion_to_euler(q)
        euler_angles.append([roll, pitch, yaw])
    euler_angles = np.array(euler_angles)

    # 2. 角速度 (gyro)
    gyro = df[['_sensor_combined_0_gyro_rad[0]',
               '_sensor_combined_0_gyro_rad[1]',
               '_sensor_combined_0_gyro_rad[2]']].values

    # 3. 速度
    vel = df[['_vehicle_local_position_0_vx',
              '_vehicle_local_position_0_vy',
              '_vehicle_local_position_0_vz']].values

    # 4. 加速度
    accel = df[['_sensor_combined_0_accelerometer_m_s2[0]',
                '_sensor_combined_0_accelerometer_m_s2[1]',
                '_sensor_combined_0_accelerometer_m_s2[2]']].values

    # 合并为 12 通道: [roll, pitch, yaw, gx, gy, gz, vx, vy, vz, ax, ay, az]
    data = np.concatenate([euler_angles, gyro, vel, accel], axis=1)

    # 提取故障标签
    if is_simulation:
        # SIL/HIL: 使用 fault_state 列
        fault_state = df['UAVState_data_fault_state'].values
        # 如果任何时刻有故障，标记为故障
        label = 1 if np.any(fault_state > 0) else 0
    else:
        # Real: 使用 rfly_ctrl_lxl 的 id
        if 'rfly_ctrl_lxl_slash_send_data_id' in df.columns:
            fault_ids = df['rfly_ctrl_lxl_slash_send_data_id'].values
            # 故障 ID 范围: 123450-123459, 123540-123549
            is_fault = ((fault_ids >= 123450) & (fault_ids <= 123459)) | \
                       ((fault_ids >= 123540) & (fault_ids <= 123549))
            label = 1 if np.any(is_fault) else 0
        else:
            # 从文件名推断 (Case_3000000xxx = Real motor = fault)
            filename = Path(csv_path).stem
            if filename.startswith('Case_300') or filename.startswith('Case_100'):
                label = 1  # motor fault
            else:
                label = 0  # no fault

    metadata = {
        'filename': Path(csv_path).name,
        'is_simulation': is_simulation,
        'length': len(data),
        'duration_sec': len(data) / 50.0  # 50Hz
    }

    return data, label, metadata

def create_sliding_windows(data, label, window_size=150, stride=50):
    """
    创建滑动窗口

    参数:
        data: (T, 12) 时间序列数据
        label: 文件级标签
        window_size: 窗口大小 (150 = 3秒 @ 50Hz)
        stride: 步长

    返回:
        windows: (N, 12, 150) numpy array
        labels: (N,) numpy array - 每个窗口的标签
    """
    T, C = data.shape
    windows = []
    labels = []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        window = data[start:end, :].T  # (12, 150)
        windows.append(window)
        labels.append(label)

    if len(windows) == 0:
        return None, None

    return np.array(windows), np.array(labels)

def load_rflymad_dataset(data_dir, window_size=150, stride=50):
    """
    加载整个 RflyMAD 数据集

    返回:
        X: (N, 12, 150) numpy array
        y: (N,) numpy array
        file_labels: list - 每个窗口对应的文件名
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob('Case_*.csv'))

    print(f"找到 {len(csv_files)} 个 CSV 文件")

    all_windows = []
    all_labels = []
    file_labels = []

    for csv_file in csv_files:
        try:
            data, label, metadata = load_rflymad_case(csv_file)
            windows, labels = create_sliding_windows(data, label, window_size, stride)

            if windows is not None:
                all_windows.append(windows)
                all_labels.append(labels)
                file_labels.extend([metadata['filename']] * len(windows))

                print(f"✓ {metadata['filename']}: {len(windows)} 窗口, "
                      f"标签={label}, 时长={metadata['duration_sec']:.1f}s")
        except Exception as e:
            print(f"✗ {csv_file.name}: 加载失败 - {e}")

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"\n数据集统计:")
    print(f"  总窗口数: {len(X)}")
    print(f"  正常: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"  故障: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"  数据形状: {X.shape}")

    return X, y, file_labels

if __name__ == "__main__":
    # 测试加载
    data_dir = "/home/jack/RflyMAD_processed"
    X, y, file_labels = load_rflymad_dataset(data_dir, window_size=150, stride=50)

    print(f"\n示例数据:")
    print(f"  X[0] shape: {X[0].shape}")
    print(f"  X[0] 统计: min={X[0].min():.3f}, max={X[0].max():.3f}, mean={X[0].mean():.3f}")
    print(f"  y[0]: {y[0]}")
