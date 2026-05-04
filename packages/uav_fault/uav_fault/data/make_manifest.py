"""
扫描 data/processed 中的 Case_*.csv，生成 RflyMAD 四分类 manifest。
输出: data/processed/manifest.csv
"""
import argparse
import csv
import re
from pathlib import Path

CASE_RE = re.compile(r'^Case_(\d+)\.csv$')
PLATFORM_MAP = {1: 'SIL', 2: 'HIL', 3: 'Real'}
FLIGHT_MAP = {0: 'hover', 1: 'waypoint', 2: 'velocity', 3: 'circling', 4: 'acce', 5: 'dece'}
FAULT_MAP = {
    0: 'motor', 1: 'propeller', 2: 'low_voltage', 3: 'wind_affect', 4: 'load_lose',
    5: 'accelerometer', 6: 'gyroscope', 7: 'magnetometer', 8: 'barometer', 9: 'GPS', 10: 'no_fault',
}
SENSOR_FAULTS = {'accelerometer', 'gyroscope', 'magnetometer', 'barometer', 'GPS'}
LABEL_ID_MAP = {'normal': 0, 'motor': 1, 'propeller': 2, 'sensor': 3}


def parse_case_filename(fname: str):
    m = CASE_RE.match(fname)
    if not m:
        return None
    case_id = int(m.group(1))
    data_type = case_id // 1_000_000_000
    flight_id = (case_id % 1_000_000_000) // 100_000_000
    fault_id = (case_id % 100_000_000) // 1_000_000
    case_num = case_id % 1_000_000
    return {
        'case_id': case_id,
        'platform': PLATFORM_MAP.get(data_type),
        'flight_mode': FLIGHT_MAP.get(flight_id),
        'fault_subtype': FAULT_MAP.get(fault_id),
        'case_num': case_num,
    }


def map_fault_family(fault_subtype: str) -> str:
    if fault_subtype == 'no_fault':
        return 'normal'
    if fault_subtype in {'motor', 'propeller'}:
        return 'actuator'
    if fault_subtype in SENSOR_FAULTS:
        return 'sensor'
    return 'excluded'


def map_label_4class(fault_subtype: str):
    if fault_subtype == 'no_fault':
        return 'normal'
    if fault_subtype == 'motor':
        return 'motor'
    if fault_subtype == 'propeller':
        return 'propeller'
    if fault_subtype in SENSOR_FAULTS:
        return 'sensor'
    return None


def count_rows(csv_path: Path) -> int:
    with open(csv_path, 'r') as f:
        return max(sum(1 for _ in f) - 1, 0)


def build_manifest(csv_dir: str, output_path: str):
    csv_dir = Path(csv_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_path in sorted(csv_dir.glob('Case_*.csv')):
        parsed = parse_case_filename(csv_path.name)
        if parsed is None:
            continue

        label = map_label_4class(parsed['fault_subtype'])
        if label is None or parsed['platform'] is None or parsed['flight_mode'] is None:
            continue

        rows.append({
            'filename': csv_path.name,
            'platform': parsed['platform'],
            'flight_mode': parsed['flight_mode'],
            'fault_subtype': parsed['fault_subtype'],
            'fault_family': map_fault_family(parsed['fault_subtype']),
            'label': label,
            'label_id': LABEL_ID_MAP[label],
            'case_id': parsed['case_id'],
            'case_num': parsed['case_num'],
            'group_id': csv_path.name.replace('.csv', ''),
            'n_rows': count_rows(csv_path),
        })

    if not rows:
        raise RuntimeError('没有可用的 RflyMAD processed CSV，无法生成 manifest')

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    label_counts = Counter(r['label'] for r in rows)
    platform_counts = Counter(r['platform'] for r in rows)
    print(f'样本清单已生成: {output_path}')
    print(f'总样本数: {len(rows)}')
    print('平台分布:')
    for k, v in sorted(platform_counts.items()):
        print(f'  {k}: {v}')
    print('四分类分布:')
    for k, v in sorted(label_counts.items()):
        print(f'  {k}: {v}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成 RflyMAD 四分类 manifest')
    parser.add_argument('--csv-dir', default='data/processed')
    parser.add_argument('--output', default='data/processed/manifest.csv')
    args = parser.parse_args()
    build_manifest(args.csv_dir, args.output)
