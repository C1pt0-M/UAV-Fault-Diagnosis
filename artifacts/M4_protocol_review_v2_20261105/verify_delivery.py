"""Read-only preservation and static publication checks, no original data."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
REPORT=ROOT/'docs/research/M4协议修订与无训练补充审查_v2_2026-11-05.md'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def js(name,value):
    with (OUT/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    before=json.loads((OUT/'initial_snapshot.json').read_text(encoding='utf-8'))
    after={name:sha(ROOT/name) for name in before['protected']}
    assert after==before['protected']
    head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
    status=subprocess.check_output(['git','-c','core.quotepath=false','status','--short']).decode('utf-8')
    assert head==before['head'] and status==before['status']
    assert subprocess.check_output(['git','diff','--cached','--name-only','-z'])==b''
    assert json.loads((OUT/'independent_verification.json').read_text(encoding='utf-8'))['status'].startswith('PASS_')
    completion=json.loads((OUT/'completion.json').read_text(encoding='utf-8'))
    assert completion['code_sha256']==sha(OUT/'analyze_v2_parallel.py')
    assert len(completion['fold_population_jobs'])==9
    # Summary population/gate counts against every completed query table.
    coverage=pd.read_csv(OUT/'joint_gate_coverage.csv')
    for pop in ['D1','D2','D3']:
        for fold in range(3):
            rr=pd.read_csv(OUT/f'condition_support_queries_f{fold}_{pop}.csv.gz',dtype={'association_group':str})
            for role in ['inner_fit','inner_dev','outer_held_out']:
                p=rr[rr.role.eq(role)]
                for gate in [c for c in rr if c.startswith('supported_')]+['ESS_only_primary','distance_only_Q99']:
                    row=coverage[coverage.population.eq(pop)&coverage.outer_fold.eq(fold)&coverage.role.eq(role)&coverage.stratum.eq('all')&coverage.gate.eq(gate)].iloc[0]
                    assert len(p)==row.queries and int(p[gate].sum())==row.supported
    scripts=[];jsons=[]
    for p in OUT.iterdir():
        assert p.is_file(),p
        if p.suffix=='.py':
            ast.parse(p.read_text(encoding='utf-8'));scripts.append(p.name)
            assert not re.search(r'^\s*(import torch|from torch|import tensorflow|from uav_fault)',p.read_text(encoding='utf-8'),re.M)
        if p.suffix=='.json':json.loads(p.read_text(encoding='utf-8'));jsons.append(p.name)
        assert p.suffix not in ['.pt','.npz','.pth','.ckpt']
    planned={(OUT/'publication_manifest.json').resolve(),(OUT/'delivery_verification.json').resolve()}
    targets=[]
    for md in [REPORT,OUT/'README.md',OUT/'support_report_tables.md']:
        text=md.read_text(encoding='utf-8')
        assert '<!-- V2_NUMERIC_SUPPORT_TABLES -->' not in text
        assert text.count('```')%2==0 and text.count('$$')%2==0
        width=None
        for line in text.splitlines():
            if line.startswith('|'):
                n=line.count('|')
                if width is not None:assert n==width,(md,line)
                width=n
            else:width=None
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if link.startswith(('http://','https://','#')):continue
            target=(md.parent/link.split('#')[0]).resolve();targets.append(target)
            assert target.exists() or target in planned,(md,link)
    ignored=subprocess.check_output(['git','check-ignore','--',str(REPORT.relative_to(ROOT)),str(OUT.relative_to(ROOT))]).decode('utf-8')
    js('delivery_verification.json',{'status':'PASS_V2_STATIC_DELIVERY_PRESERVATION','head':head,'git_status_unchanged':True,'git_status':status,'protected_files_checked':len(after),'protected_SHA_unchanged':True,'protected_manifest':'initial_snapshot.json','source_v1_only_no_original_data_or_weights':True,'no_models_forward_h_U_C_scores_GPU_server_training_Test_Real_Val_experiments':True,'all9_query_summary_counts_reconciled':True,'scripts_AST_checked':scripts,'JSON_checked':jsons,'local_links_checked':len(targets),'report':{'path':REPORT.relative_to(ROOT).as_posix(),'bytes':REPORT.stat().st_size,'sha256':sha(REPORT),'lines':len(REPORT.read_text(encoding='utf-8').splitlines())},'ignored_by_existing_rules':ignored,'staged_files':0,'commit_push':False,'scope_limit':'Static support/mathematics/protocol audit only. No claim of diagnostic efficacy, fault causality or safety.'})
    files=sorted([p for p in OUT.iterdir() if p.is_file()]+[REPORT])
    js('publication_manifest.json',{'scope':'New v2 directory and report only; historical files protected separately','exclude_self':'publication_manifest.json','base_commit':head,'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'file_count_excluding_self':len(files),'bytes_excluding_self':sum(p.stat().st_size for p in files)})
    assert all(p.exists() for p in targets)
    print(json.dumps({'status':'PASS_V2_FINAL_LINKS_AND_PRESERVATION','new_files':len(files)+1,'protected_files':len(after),'report_lines':len(REPORT.read_text(encoding='utf-8').splitlines()),'git_status_unchanged':True,'report_sha256':sha(REPORT)},ensure_ascii=False))


if __name__=='__main__':main()
