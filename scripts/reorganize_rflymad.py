#!/usr/bin/env python3
"""
RflyMAD 数据集完整重组脚本
原始结构 → 官方目录结构，输出到 /home/jack/UAV-Fault-Diagnosis/data/

目标结构:
data/
  SIL/
    hover/
      motor/
      no_fault/
      propeller/
      accelerometer/
      gyroscope/
      ...
    acce/
    circling/
    ...
  HIL/
    ...
  Real/
    ...
"""

import os
import shutil
from pathlib import Path

SRC = Path('/home/jack/RflyMAD')
DST = Path('/home/jack/UAV-Fault-Diagnosis/data')

# 我们只需要这 4 类故障
TARGET_FAULTS = {'motor', 'no_fault', 'propeller', 'accelerometer', 'gyroscope',
                 'barometer', 'GPS', 'magnetometer'}

# 传感器子类型映射: 原始名称 → 目标目录名
SENSOR_MAP = {
    'acce': 'accelerometer',
    'baro': 'barometer',
    'GPS': 'GPS',
    'gyro': 'gyroscope',
    'mag': 'magnetometer',
}

# 飞行状态列表
FLIGHT_MODES = ['acce', 'circling', 'dece', 'hover', 'velocity', 'waypoint']

stats = {'total': 0, 'skipped': 0}


def link_cases(src_dir, dst_dir):
    """用符号链接将案例目录链接到目标位置"""
    if not src_dir.exists():
        return 0
    count = 0
    for case in sorted(src_dir.iterdir()):
        if case.is_dir():
            dst_case = dst_dir / case.name
            if not dst_case.exists():
                # 使用符号链接节省磁盘空间
                os.symlink(str(case.resolve()), str(dst_case))
                count += 1
                stats['total'] += 1
    return count


def process_platform(platform):
    """处理一个平台 (SIL/HIL/Real) 的所有数据"""
    print(f"\n{'='*50}")
    print(f"处理 {platform}")
    print(f"{'='*50}")

    # --- Motor ---
    # Motor_1: hover-motor-1, hover-motor-2, hover-motor-3, circling-motor-1, ...
    # Motor_2: acce-motor, dece-motor, velocity-motor, waypoint-motor-1, ...
    # Real-Motor: acce/, circling/, hover/, ...
    motor_dirs = []
    if platform == 'Real':
        motor_dirs = [SRC / f'{platform}-Motor']
    else:
        motor_dirs = [SRC / f'{platform}_Motor_1', SRC / f'{platform}_Motor_2']

    for motor_dir in motor_dirs:
        if not motor_dir.exists():
            continue
        for sub in sorted(motor_dir.iterdir()):
            if not sub.is_dir():
                continue
            # 解析飞行状态
            name = sub.name  # e.g. "hover-motor-1" or "hover" (Real)
            flight_mode = None
            for fm in FLIGHT_MODES:
                if name.startswith(fm):
                    flight_mode = fm
                    break
            if not flight_mode:
                print(f"  [跳过] 无法解析飞行状态: {sub}")
                stats['skipped'] += 1
                continue

            dst = DST / platform / flight_mode / 'motor'
            dst.mkdir(parents=True, exist_ok=True)
            cnt = link_cases(sub, dst)
            if cnt > 0:
                print(f"  motor/{flight_mode}: +{cnt} 案例")

    # --- NoFault ---
    nf_dir = SRC / f'{platform}-NoFault'
    if not nf_dir.exists():
        nf_dir = SRC / f'{platform}-No_Fault'
    if nf_dir.exists():
        for sub in sorted(nf_dir.iterdir()):
            if not sub.is_dir():
                continue
            flight_mode = sub.name  # "hover", "acce", etc.
            dst = DST / platform / flight_mode / 'no_fault'
            dst.mkdir(parents=True, exist_ok=True)
            cnt = link_cases(sub, dst)
            if cnt > 0:
                print(f"  no_fault/{flight_mode}: +{cnt} 案例")

    # --- Propeller ---
    prop_dir = SRC / f'{platform}_Prop'
    if prop_dir.exists():
        for sub in sorted(prop_dir.iterdir()):
            if not sub.is_dir():
                continue
            name = sub.name  # "hover-prop"
            flight_mode = None
            for fm in FLIGHT_MODES:
                if name.startswith(fm):
                    flight_mode = fm
                    break
            if not flight_mode:
                stats['skipped'] += 1
                continue

            dst = DST / platform / flight_mode / 'propeller'
            dst.mkdir(parents=True, exist_ok=True)
            cnt = link_cases(sub, dst)
            if cnt > 0:
                print(f"  propeller/{flight_mode}: +{cnt} 案例")

    # --- Sensors (accelerometer, barometer, GPS, gyroscope, magnetometer) ---
    sensor_dir = SRC / f'{platform}-Sensors'
    if sensor_dir.exists():
        for sub in sorted(sensor_dir.iterdir()):
            if not sub.is_dir():
                continue
            name = sub.name  # "hover-gyro", "acce-acce", "hover-baro"
            flight_mode = None
            sensor_type = None
            for fm in FLIGHT_MODES:
                if name.startswith(fm + '-'):
                    flight_mode = fm
                    sensor_suffix = name[len(fm)+1:]  # "gyro", "acce", "baro", etc.
                    sensor_type = SENSOR_MAP.get(sensor_suffix)
                    break
            if not flight_mode or not sensor_type:
                print(f"  [跳过] 无法解析: {sub}")
                stats['skipped'] += 1
                continue

            dst = DST / platform / flight_mode / sensor_type
            dst.mkdir(parents=True, exist_ok=True)
            cnt = link_cases(sub, dst)
            if cnt > 0:
                print(f"  {sensor_type}/{flight_mode}: +{cnt} 案例")


if __name__ == '__main__':
    print("RflyMAD 数据集重组")
    print(f"源目录: {SRC}")
    print(f"目标目录: {DST}")
    print(f"目标故障类型: 正常 / 电机 / 螺旋桨 / 传感器(5类)")

    # 清理旧目标目录
    if DST.exists():
        print(f"\n清理旧目录: {DST}")
        shutil.rmtree(DST)

    DST.mkdir(parents=True, exist_ok=True)

    for platform in ['SIL', 'HIL', 'Real']:
        process_platform(platform)

    print(f"\n{'='*50}")
    print(f"完成！共链接 {stats['total']} 个案例，跳过 {stats['skipped']} 个")
    print(f"{'='*50}")

    # 统计结果
    print(f"\n目录结构统计:")
    for platform in ['SIL', 'HIL', 'Real']:
        pdir = DST / platform
        if not pdir.exists():
            continue
        print(f"\n  {platform}/")
        for fm in sorted(pdir.iterdir()):
            if not fm.is_dir():
                continue
            print(f"    {fm.name}/")
            for fault in sorted(fm.iterdir()):
                if not fault.is_dir():
                    continue
                case_count = sum(1 for c in fault.iterdir() if c.is_dir())
                print(f"      {fault.name}: {case_count} 案例")
