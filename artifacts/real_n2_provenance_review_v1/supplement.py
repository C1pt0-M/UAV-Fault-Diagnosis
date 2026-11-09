"""Additional provenance checks only; preserves the completed extraction."""
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import struct
import sys
import importlib.util
import numpy as np

BASE = Path('D:/UAV-Fault-Diagnosis/artifacts/real_n2_provenance_review_v1')
spec = importlib.util.spec_from_file_location('provenance_inspection', BASE/'inspect.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.OUT = BASE
if (BASE/'supplement_results.json').exists():
    raise FileExistsError('Supplement already complete')
P = BASE/'attempt3'
meta = json.loads((P/'case_records.json').read_text(encoding='utf-8'))
native = json.loads((P/'native_records.json').read_text(encoding='utf-8'))
allow = audit.eligible(list(csv.DictReader((audit.ROOT/'configs/data/real_v2_split.csv').open(encoding='utf-8-sig'))))
access = []


def read(path, uid, purpose):
    path = Path(path).resolve()
    if uid not in allow:
        raise PermissionError(uid)
    raw = (audit.RAW/uid).resolve()
    data_root = Path('D:/UAV-Data/causal_input_v1')
    allowed_paths = {(data_root/'case_records'/(uid.replace('/','__')+'.json')).resolve(),
                     (data_root/'cases'/(uid.replace('/','__')+'.npz')).resolve()}
    if not path.is_relative_to(raw) and path not in allowed_paths:
        raise PermissionError(str(path))
    data = path.read_bytes()
    access.append(dict(case_uid=uid, path=str(path), purpose=purpose, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return data


# Weather exports are provenance records, not model inputs or physical ground truth.
weather = []
for nr in native:
    uid = nr['case_uid']
    for path in (audit.RAW/uid).glob('wind*.csv'):
        data = read(path, uid, 'available weather export schema/range')
        rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
        weather.append(dict(case_uid=uid, path=str(path), rows=rows))
audit.save('weather_export_review.json', weather)

# Read ONLY saved time/label masks, never canonical_rows or features/scores.
crops = []
for uid, row in sorted(allow.items()):
    if row['family'] != 'normal':
        continue
    base = Path('D:/UAV-Data/causal_input_v1')
    rec = json.loads(read(base/'case_records'/(uid.replace('/','__')+'.json'), uid, 'normal-role crop metadata'))
    data = read(rec['frame_file'], uid, 'saved crop timestamps and reference/mask ONLY')
    if hashlib.sha256(data).hexdigest() != rec['output_identities'][rec['frame_file']]:
        raise ValueError('Saved normal crop identity changed')
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        times, state, valid = z['timestamp_us'], z['reference_state'], z['row_label_valid']
    nr = next(x for x in native if x['case_uid'] == uid)
    first = nr['first_active_received']
    crops.append(dict(case_uid=uid, role=row['role_v1'], first_saved_us=int(times[0]), last_saved_us=int(times[-1]),
        rows=len(times), saved_valid_rows=int(valid.sum()), saved_valid_reference_active=int(((state==1)&valid).sum()),
        original_ULog_first_active_us=int(first['timestamp']) if first else None,
        original_received_first_after_saved_end=bool(first['timestamp']>times[-1]) if first else None,
        saved_end_to_received_first_s=(first['timestamp']-int(times[-1]))/1e6 if first else None))
audit.table('normal_crop_boundaries.csv', crops)

# Native original bag 124_1: existing decoder, strict indexed record and payload checks.
# Append tool-local packages, so installed NumPy is never shadowed or upgraded.
sys.path.append('D:/tmp/real_labeltime_tools_20261031')
from rosbags.rosbag1 import Reader
from rosbags.rosbag1.reader import Header, RecordType
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
uid = 'REAL/waypoint/motor/124_1'
path = audit.only((audit.RAW/uid).glob('**/*.bag'))
assert uid in allow and path.resolve().is_relative_to((audit.RAW/uid).resolve())
before = (path.stat().st_size, path.stat().st_mtime_ns)
read(path, uid, 'original 124_1 bag identity before strict decode')
targets = ['/mavros/rfly_ctrl_lxl/send_data', '/mavros/offb_flight_mode/send_data', '/mavros/timesync_status', '/rosout']
store = get_typestore(Stores.ROS1_NOETIC)
messages = {k: [] for k in targets}
with Reader(path) as reader, path.open('rb') as file:
    conns = {c.id: c for c in reader.connections if c.topic in targets}
    if set(c.topic for c in conns.values()) != set(targets):
        raise ValueError('Missing required topic')
    for c in conns.values():
        store.register(get_types_from_msg(c.msgdef.data, c.msgtype))
    expected = sum(len(reader.indexes[cid]) for cid in conns)
    decoded = 0
    for position, chunk in reader.chunks.items():
        entries = [(e.offset, cid, e.time) for cid in conns for e in reader.indexes[cid] if e.chunk_pos == position]
        if not entries:
            continue
        file.seek(chunk.datapos)
        packed = file.read(chunk.datasize)
        if len(packed) != chunk.datasize:
            raise ValueError('Truncated chunk')
        raw = chunk.decompressor(packed)
        stream = io.BytesIO(raw)
        for offset, cid, ns in sorted(entries):
            stream.seek(offset)
            header = Header.read(stream)
            if (header.get_uint8('op') != RecordType.MSGDATA or header.get_uint32('conn') != cid
                    or int(header.get_time('time')) != ns):
                raise ValueError('Index/header disagreement')
            sizebytes = stream.read(4)
            if len(sizebytes) != 4:
                raise ValueError('Truncated payload size')
            size = struct.unpack('<I', sizebytes)[0]
            if size > len(raw)-stream.tell():
                raise ValueError('Payload exceeds chunk')
            payload = stream.read(size)
            msg = store.deserialize_ros1(payload, conns[cid].msgtype)
            if bytes(store.serialize_ros1(msg, conns[cid].msgtype)) != payload:
                raise ValueError('Payload roundtrip mismatch')
            topic = conns[cid].topic
            item = dict(rosbagTimestamp=int(ns))
            if topic == targets[0]:
                item.update(index=int(msg.index), id=int(msg.id), mode=int(msg.mode), controls=[float(x) for x in msg.controls])
            elif topic == targets[1]:
                item.update(program_time=float(msg.program_time), sleep_status=int(msg.sleep_status), flight_status=int(msg.flight_status))
            elif topic == targets[2]:
                item['remote_timestamp_ns'] = int(msg.remote_timestamp_ns)
            else:
                item.update(name=msg.name, file=msg.file, function=msg.function, line=int(msg.line), level=int(msg.level), msg=msg.msg)
            messages[topic].append(item)
            decoded += 1
if decoded != expected or before != (path.stat().st_size, path.stat().st_mtime_ns):
    raise ValueError('Incomplete or changed original bag')
for topic, rows in messages.items():
    if any(a['rosbagTimestamp']>b['rosbagTimestamp'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Physical record clock reversal in '+topic)
export = audit.only((audit.RAW/uid).glob('**/_slash_mavros_slash_rfly_ctrl_lxl_slash_send_data.csv'))
rows = list(csv.DictReader(io.StringIO(read(export, uid, '124_1 bag vs export exact full command fields').decode('utf-8-sig'))))
export_commands = [dict(rosbagTimestamp=int(r['rosbagTimestamp']), index=int(r['index']), id=int(r['id']), mode=int(r['mode']),
                        controls=__import__('ast').literal_eval(r['controls'])) for r in rows]
if export_commands != messages[targets[0]]:
    raise ValueError('Original full commands differ from exported CSV')
last, transitions = None, []
for r in messages[targets[0]]:
    key = (r['index'],r['id'],r['mode'],tuple(r['controls']))
    if key != last:
        transitions.append(r)
        last = key
first_sync = messages[targets[2]][0]
first = next(r for r in messages[targets[0]] if 123450 <= r['id'] <= 123456)
pg = [r for r in messages[targets[1]] if r['rosbagTimestamp'] <= first['rosbagTimestamp']][-1]
# Exactly the historical first-timesync anchor; no fit, new offset, or relabel.
mapped_us = first_sync['remote_timestamp_ns']//1000 + (first['rosbagTimestamp']-first_sync['rosbagTimestamp'])/1000
active = [r for r in transitions if 123450 <= r['id'] <= 123456]
active_ns = [r['rosbagTimestamp'] for r in active]
periods = [(b-a)/1e9 for a,b in zip(active_ns,active_ns[1:])]
lengths = []
for i,r in enumerate(transitions[:-1]):
    if 123450 <= r['id'] <= 123456:
        lengths.append((transitions[i+1]['rosbagTimestamp']-r['rosbagTimestamp'])/1e9)
audit.save('124_1_original_bag_check.json', dict(status='STRICT_INDEXED_FIELDS_VERIFIED', bag_path=str(path),
    indexed_target_records=expected, decoded_target_records=decoded, topics={k:len(v) for k,v in messages.items()},
    full_command_export_exact=True, numpy_version=np.__version__, reader_version=__import__('importlib.metadata').metadata.version('rosbags'),
    package_resolution='existing tool path appended; no installs/upgrades', first_sync=first_sync,
    first_active_sent=first, program_clock_last_before_first_send=pg,
    first_sent_mapped_by_existing_anchor_us=mapped_us,
    state_transitions=transitions, active_intervals=len(active), complete_active_durations_s=lengths,
    active_start_periods_s=periods,
    rosout_source_files=sorted({r['file'] for r in messages['/rosout']}),
    rosout_native_messages=messages['/rosout'], original_bag_unchanged=True,
    identity_read_registered_in_input_log=True, decoder_additional_known_opens=['Reader(path)', 'path.open(rb)']))
audit.save('supplement_input_access.json', access)
audit.save('supplement_results.json', dict(status='COMPLETE_READ_ONLY_SUPPLEMENT', weather_cases=len(weather),
    normal_crops=len(crops), bag_cases=1, material_identity_reads=len(access),
    no_models=True, no_features_loaded=True, no_new_labels=True, no_new_clock_fit=True))
print('SUPPLEMENT_COMPLETE',len(access),len(weather),len(crops),decoded)
