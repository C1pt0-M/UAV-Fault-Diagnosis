"""Verify original Real bag command/sync records; never repair a damaged bag.

Uses installed rosbags in the configured tool directory (not project deps).
Read each indexed target record with header/index/length/type checks. Failed
records are evidence, not skipped into a supposedly complete reference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import struct
import sys

import numpy as np

from audit_rflymad_raw_sources import digest

ROOT = Path(__file__).resolve().parents[2]
COMMAND = '/mavros/rfly_ctrl_lxl/send_data'
SYNC = '/mavros/timesync_status'


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def compare_export(path, expected, columns):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    actual = [tuple(int(row[k]) for k in columns) for row in rows]
    differences = [{'row': i, 'bag': list(a), 'export': list(b)}
                   for i, (a, b) in enumerate(zip(expected, actual)) if a != b]
    return {'exact': len(actual) == len(expected) and not differences,
            'bag_rows': len(expected), 'export_rows': len(actual),
            'different_rows': len(differences), 'examples': differences[:10]}


def verify_case(task):
    case, inventory, dest = task
    result = {'case_uid': case['case_uid'], 'role': case['role'],
              'status': 'QUARANTINE', 'errors': [], 'export_differences': {}, 'warnings': []}
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.rosbag1.reader import Header, RecordType
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg
        bags = inventory['sources']['bag']
        if len(bags) != 1:
            raise ValueError('Expected exactly one original bag')
        path = Path(bags[0])
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        result.update(bag_path=str(path), bag_sha256=digest(path))
        command, sync, order = [], [], []
        store = get_typestore(Stores.ROS1_NOETIC)
        with Reader(path) as reader, path.open('rb') as file:
            connections = {c.id: c for c in reader.connections if c.topic in [COMMAND, SYNC]}
            if Counter(c.topic for c in connections.values()) != {COMMAND: 1, SYNC: 1}:
                raise ValueError('Expected one connection for each required topic')
            for connection in connections.values():
                store.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
            expected = sum(len(reader.indexes[cid]) for cid in connections)
            decoded = 0
            for position, chunk in reader.chunks.items():
                entries = [(entry.offset, cid, entry.time) for cid in connections
                           for entry in reader.indexes[cid] if entry.chunk_pos == position]
                if not entries:
                    continue
                file.seek(chunk.datapos)
                packed = file.read(chunk.datasize)
                if len(packed) != chunk.datasize:
                    raise ValueError('Truncated chunk bytes')
                raw = chunk.decompressor(packed)
                stream = io.BytesIO(raw)
                # Physical record order is explicit. Never silently sort malformed
                # timestamps into an apparently valid event trace.
                for offset, cid, indexed_ns in sorted(entries):
                    context = {'chunk': position, 'offset': offset, 'topic': connections[cid].topic,
                               'index_timestamp_ns': int(indexed_ns)}
                    try:
                        stream.seek(offset)
                        header = Header.read(stream)
                        if header.get_uint8('op') != RecordType.MSGDATA:
                            raise ValueError('Index does not point to a message record')
                        if header.get_uint32('conn') != cid:
                            raise ValueError('Index connection differs from message header')
                        timestamp = int(header.get_time('time'))
                        if timestamp != indexed_ns:
                            raise ValueError('Index timestamp differs from message header: ' + str(timestamp))
                        sizebytes = stream.read(4)
                        if len(sizebytes) != 4:
                            raise ValueError('Truncated message size')
                        size = struct.unpack('<I', sizebytes)[0]
                        if size > len(raw) - stream.tell():
                            raise ValueError('Message payload extends beyond chunk')
                        payload = stream.read(size)
                        message = store.deserialize_ros1(payload, connections[cid].msgtype)
                        # Exact roundtrip checks field alignment, complete payload
                        # length and serialization identity, including custom types.
                        if bytes(store.serialize_ros1(message, connections[cid].msgtype)) != payload:
                            raise ValueError('Message payload does not roundtrip exactly')
                        if connections[cid].topic == COMMAND:
                            command.append((timestamp, int(message.id)))
                        else:
                            sync.append((timestamp, int(message.remote_timestamp_ns)))
                        order.append((offset, timestamp))
                        decoded += 1
                    except Exception as exc:
                        result['errors'].append(dict(context, error=type(exc).__name__ + ': ' + str(exc)))
            result.update(indexed_target_records=expected, decoded_target_records=decoded)
            if decoded != expected:
                result['errors'].append({'error': 'Not every indexed target record was decoded'})
        result.update(command_rows=len(command), sync_rows=len(sync))
        if not command or not sync:
            raise ValueError('Missing complete command/sync trace')
        ct = np.asarray([x[0] for x in command], dtype=np.int64)
        st = np.asarray([x[0] for x in sync], dtype=np.int64)
        remote = np.asarray([x[1] for x in sync], dtype=np.int64)
        if (np.diff(ct) < 0).any() or (np.diff(st) < 0).any():
            result['errors'].append({'error': 'Original target-topic record clock reversal'})
        if remote[0] <= 0 or (np.diff(remote) < 0).any():
            result['errors'].append({'error': 'Original timesync remote-clock reversal or invalid first anchor'})
        result['export_differences']['command'] = compare_export(
            case['ros_command_export'], command, ['rosbagTimestamp', 'id'])
        result['export_differences']['sync'] = compare_export(
            case['ros_timesync_export'], sync, ['rosbagTimestamp', 'remote_timestamp_ns'])
        # Any mismatch in required fields stays unresolved; don't choose the source
        # that happens to agree better with a model or a prior label.
        if not all(x['exact'] for x in result['export_differences'].values()):
            result['errors'].append({'error': 'Original required bag fields differ from existing export'})
        if ct[0] != st[0]:
            result['warnings'].append('Topics start at distinct times; no extrapolation before/after commands')
        if (path.stat().st_size, path.stat().st_mtime_ns) != before:
            raise ValueError('Original bag changed during read')
        result['original_bag_unchanged'] = True
        if not result['errors']:
            # Exact original data, not a reconstructed state series.
            filename = 'reference_' + case['case_uid'].replace('/', '__') + '.npz'
            np.savez_compressed(dest / filename, command_ros_ns=ct,
                command_id=np.asarray([x[1] for x in command], dtype=np.int64),
                sync_ros_ns=st, sync_remote_ns=remote)
            result.update(status='VERIFIED_REQUIRED_BAG_FIELDS', reference_file=filename,
                          reference_sha256=digest(dest / filename))
    except Exception as exc:
        result['errors'].append({'error': type(exc).__name__ + ': ' + str(exc)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tools', type=Path, default=Path('D:/tmp/real_labeltime_tools_20261031'))
    args = parser.parse_args()
    # Append, do not shadow installed numpy/pandas/torch with tool-local deps.
    sys.path.append(str(args.tools.resolve()))
    import importlib.metadata
    reader_version = importlib.metadata.version('rosbags')
    audit = ROOT / 'artifacts/real_labeltime_audit_20261031'
    cases = [json.loads(s) for s in (audit / 'case_results.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(cases) != 465 or any(c['status'] != 'diagnostic_complete' for c in cases):
        raise ValueError('Expected complete 465-case audit')
    inventory = {r['case_uid'].lower(): r for r in
                 (json.loads(s) for s in (ROOT / 'artifacts/raw_source_audit_20261030/case_inventory.jsonl').read_text(encoding='utf-8').splitlines())
                 if r['domain'] == 'Real'}
    output = args.output.resolve()
    if output.is_relative_to(Path('D:/RflyMAD').resolve()) or output.is_relative_to(Path('D:/UAV-Data').resolve()):
        raise ValueError('Bag verification outputs belong in separate artifacts')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'plan.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': digest(__file__), 'reader': 'rosbags ' + reader_version,
        'cases_sha256': digest(audit / 'case_results.jsonl'),
        'policy': 'Strict original indexed target record checks. Quarantine failures, no resync, no timestamp/clock guessing. Verify only required command timestamp/ID and sync timestamp/remote fields, not entire flight payload integrity.'})
    tasks = [(c, inventory[c['case_uid'].lower()], output) for c in cases]
    results = []
    with (output / 'case_results.jsonl').open('x', encoding='utf-8') as stream:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i, result in enumerate(pool.map(verify_case, tasks), 1):
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                results.append(result)
                if i % 100 == 0:
                    print('ORIGINAL_BAG_VERIFIED', i, '/465', flush=True)
    summary = {'cases': len(results), 'statuses': dict(Counter(r['status'] for r in results)),
        'quarantined': [{'case_uid': r['case_uid'], 'role': r['role'], 'errors': r['errors']}
                        for r in results if r['status'] != 'VERIFIED_REQUIRED_BAG_FIELDS']}
    save_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
