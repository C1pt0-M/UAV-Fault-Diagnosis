"""Final static delivery check; no model construction or forward.

Reads only the explicitly allowed source Train files, protected workspace
files, owning state source functions and new delivery assets. New records
use exclusive-create; this script must not overwrite prior evidence.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPORT = ROOT / 'docs/research/M4专项研究设计与数据可行性报告_v1_2026-11-05.md'
DATA = Path('D:/UAV-Data/causal_input_v1')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def save(name, value):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    assert not (OUT / 'delivery_verification.json').exists()
    assert not (OUT / 'publication_manifest.json').exists()
    initial = load('initial_workspace_snapshot.json')
    status = subprocess.check_output(['git', '-c', 'core.quotepath=false', 'status', '--short'], cwd=ROOT).decode('utf-8')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    assert status.replace('\r\n', '\n') == initial['status'].replace('\r\n', '\n')
    assert head == initial['head']
    protected = {name: sha(ROOT / name) for name in initial['preserved_files']}
    assert protected == initial['preserved_files']
    identity = load('input_identity.json')
    train_sha = {name: sha(DATA / name) for name in identity['inputs']}
    assert train_sha == identity['inputs']
    assert sha(OUT / 'analysis_plan.json') == identity['plan_sha256']
    assert sha(OUT / 'analyze_train.py') == identity['script_sha256']
    state = load('state_rule_identity.json')
    for relative, name, key in [
        ('scripts/research/collect_m3_formal_results.py', 'observed_states', 'vector_function_sha256'),
        ('scripts/research/analyze_13_feature_screening.py', 'observed_motion', 'scalar_function_sha256')]:
        source = (ROOT / relative).read_text(encoding='utf-8-sig')
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert hashlib.sha256(ast.get_source_segment(source, function).encode()).hexdigest() == state[key]

    frame = pd.read_csv(OUT / 'window_state_ledger.csv.gz', dtype={'association_group': str}, keep_default_na=False)
    assert len(frame) == 162069 and frame.original_cache_index.is_unique
    assert frame.case_uid.nunique() == 2570 and frame.association_group.nunique() == 1296
    assert frame.known_normal.sum() == 83051 and frame.qualified_y.eq(1).sum() == 79018
    assert frame.pure_fault.sum() == 73098
    assert not frame.loc[frame.state.eq('descent_proxy'), 'known_normal'].any()
    assert frame.state.eq('descent_proxy').sum() == 1344
    for k, failed in [(5, 167), (3, 87)]:
        design = pd.read_csv(OUT / f'fold_assignments_{k}.csv', dtype={'association_group': str})
        assert design.association_group.is_unique and len(design) == 1296
        assigned = frame.association_group.map(design.set_index('association_group').outer_fold)
        assert assigned.notna().all()
        assert pd.DataFrame({'case': frame.case_uid, 'fold': assigned}).groupby('case').fold.nunique().eq(1).all()
        assert sum(assigned.eq(i).sum() for i in range(k)) == 162069
        support = pd.read_csv(OUT / f'fold_state_fault_domain_support_{k}.csv')
        # Column names come from the frozen descriptive writer, not a model.
        flag = 'heldout_descriptive_pass'
        if flag not in support.columns:
            flag = next(c for c in support.columns if c.endswith('pass') and 'held' in c)
        assert (~support[flag]).sum() == failed
    nested = pd.read_csv(OUT / 'nested_group_assignments_3.csv', dtype={'association_group': str})
    for outer in range(3):
        part = nested[nested.outer_evaluation_fold.eq(outer)]
        assert part.association_group.is_unique and len(part) == 1296
        roles = frame.association_group.map(part.set_index('association_group').role)
        assert roles.notna().all()
        assert pd.DataFrame({'case': frame.case_uid, 'role': roles}).groupby('case').role.nunique().eq(1).all()
    assert load('independent_verification.json')['status'].startswith('PASS_')
    assert load('nested_design_verification.json')['case_or_group_crossings'] == 0

    text = REPORT.read_text(encoding='utf-8')
    assert text.count('```mermaid') == 3
    assert len(re.findall(r'^## \d+\.', text, re.M)) == 15
    assert text.count('```') % 2 == 0 and text.count('$$') % 2 == 0
    links = re.findall(r'\]\(([^)]+)\)', text) + re.findall(r'^\[[^\]]+\]:\s*(\S+)', text, re.M)
    planned = {(OUT / 'delivery_verification.json').resolve(), (OUT / 'publication_manifest.json').resolve()}
    targets = [(REPORT.parent / link.split('#')[0]).resolve() for link in links if not link.startswith(('https://', 'http://', '#'))]
    missing = [str(path) for path in targets if not path.exists() and path not in planned]
    assert not missing, missing
    definitions = set(re.findall(r'^\[([^\]]+)\]:', text, re.M))
    references = set(re.findall(r'\]\[([^\]]+)\]', text))
    assert references <= definitions
    # Verify table shape and parse new scripts statically without executing them.
    table_width = None
    for line in text.splitlines():
        if line.startswith('|'):
            width = line.count('|')
            if table_width is not None:
                assert width == table_width, line
            table_width = width
        else:
            table_width = None
    json_count, python_count = 0, 0
    for file in OUT.iterdir():
        if file.suffix == '.json':
            json.loads(file.read_text(encoding='utf-8-sig'))
            json_count += 1
        if file.suffix == '.py':
            ast.parse(file.read_text(encoding='utf-8-sig'))
            python_count += 1
        assert file.suffix not in ['.pt', '.npz', '.pth', '.ckpt'], file
    ignored = subprocess.check_output(['git', 'check-ignore', '--', str(REPORT.relative_to(ROOT)), str(OUT.relative_to(ROOT))], cwd=ROOT).decode('utf-8')
    save('delivery_verification.json', {
        'status': 'PASS_STATIC_DELIVERY_AND_PRESERVATION', 'git_head': head,
        'git_status_unchanged': True, 'git_status': status,
        'preserved_file_sha256': protected, 'source_Train_allowlist_SHA_unchanged': train_sha,
        'frozen_state_functions_and_analysis_plan_script_unchanged': True,
        'reconciled_population': {'windows': 162069, 'cases': 2570, 'association_groups': 1296, 'normal': 83051, 'fault_endpoint': 79018, 'pure_fault': 73098},
        'fold_and_nested_case_group_crossings': 0, 'state_cell_failures': {'5fold': 167, '3fold': 87},
        'report': {'path': str(REPORT.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha(REPORT), 'bytes': REPORT.stat().st_size,
                   'lines': len(text.splitlines()), 'chapters': 15, 'Mermaid': 3, 'local_links': len(targets),
                   'link_check': 'All existing or exact two final records created by this script; post-write existence asserted'},
        'JSON_parsed_before_final_records': json_count, 'new_Python_files_AST_parsed': python_count,
        'new_files_ignored_by_existing_rules': ignored,
        'scope': 'No training,model construction/forward/GPU/server/Test/Real/new Val analysis,installation,historical mutation,commit or push. Data-only Train tensor statistics were performed earlier.',
        'limits': 'Descriptive and protocol correctness, not predictive efficacy or causal/real-flight validation. Protected-file/SHA scope is explicit; no prohibited data read for a global hash.'})
    files = [p for p in OUT.iterdir() if p.is_file()] + [REPORT]
    manifest = {'scope': 'Only newly created M4 report and independent preflight directory',
                'exclude_self_to_avoid_circular_hash': 'publication_manifest.json',
                'git_head': head, 'files': [{'path': str(p.relative_to(ROOT)).replace('\\', '/'), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(files)],
                'file_count_excluding_self': len(files), 'total_bytes_excluding_self': sum(p.stat().st_size for p in files)}
    save('publication_manifest.json', manifest)
    assert all(p.exists() for p in targets)
    assert status == subprocess.check_output(['git', '-c', 'core.quotepath=false', 'status', '--short'], cwd=ROOT).decode('utf-8')
    print(json.dumps({'status': 'PASS_FINAL_LINKS_SHA_AND_GIT', 'report_lines': len(text.splitlines()),
                      'new_files_including_manifest': len(files) + 1,
                      'new_bytes_including_manifest': manifest['total_bytes_excluding_self'] + (OUT / 'publication_manifest.json').stat().st_size,
                      'all_local_links_resolve': True, 'protected_files_SHA_unchanged': True, 'source_Train_SHA_unchanged': True,
                      'git_status_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
