"""PI supplement 2: repeatable, CPU-only B1 recovery. Numerical code stays in B1.

Use: python -X utf8 -E -P -B scripts/research/resume_real_v2_b1_learning_curve.py
Existing models/OOF are trusted from the approved recent byte audit, not rehashed.
New models embed their fit receipt in one atomic checkpoint. Predictions/features
and completed analysis stages are cached; a reboot never requires a completed fit
again. Partial temporary files are retained, never mistaken for committed output.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import io
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/research'), str(ROOT/'packages/uav_fault')]
import numpy as np
import pandas as pd
import importlib.metadata
from threadpoolctl import threadpool_limits
import real_v2_b1_learning_curve as b

OUT, LOCAL, DATA = b.OUT, b.LOCAL, b.DATA
IO_EVENTS = []


def atomic_write(path, writer, replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:raise FileExistsError(str(path))
    with tempfile.NamedTemporaryFile(mode='w+b',prefix=path.name+'.',suffix='.partial',dir=path.parent,delete=False) as f:
        pending=Path(f.name);writer(f);f.flush();os.fsync(f.fileno())
    if path.exists() and not replace:raise FileExistsError(str(path))
    for attempt in range(3):
        try:
            os.replace(pending,path)
            break
        except PermissionError as exc:
            event={'operation':'os.replace','target':str(path),'attempt':attempt+1,'error':repr(exc)}
            IO_EVENTS.append(event);print('ATOMIC_REPLACE_PERMISSION',json.dumps(event,ensure_ascii=False),flush=True)
            if attempt==2:raise
            time.sleep(.1 if attempt==0 else .25)


def write_json(path,value,replace=False):
    payload=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    atomic_write(path,lambda f:f.write(payload),replace)


def complete_jsonl_bytes(path):
    path=Path(path);raw=path.read_bytes() if path.exists() else b''
    lines=raw.splitlines(keepends=True)
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError):
            if i!=len(lines)-1 or raw.endswith(b'\n'):raise ValueError('Corrupt complete journal record: '+str(path))
            # Preserve the interrupted original before repairing only its tail.
            with tempfile.NamedTemporaryFile(mode='wb',prefix=path.name+'.',suffix='.interrupted',dir=path.parent,delete=False) as f:
                f.write(raw);f.flush();os.fsync(f.fileno());backup=f.name
            prefix=b''.join(lines[:i])
            atomic_write(path,lambda f:f.write(prefix),replace=True)
            event={'operation':'preserve_truncated_journal_tail','target':str(path),'backup':backup}
            IO_EVENTS.append(event);print('JOURNAL_TAIL_PRESERVED',json.dumps(event,ensure_ascii=False),flush=True)
            return prefix
    return raw


def append_jsonl(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);old=complete_jsonl_bytes(path)
    payload=(b'\n' if old and not old.endswith(b'\n') else b'')+json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')+b'\n'
    # Appending avoids Windows FILE_SHARE_DELETE contention on an active journal.
    with path.open('ab') as f:f.write(payload);f.flush();os.fsync(f.fileno())


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def read_jsonl(path):
    return [json.loads(s) for s in complete_jsonl_bytes(path).splitlines() if s.strip()]


def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def write_csv(path,rows):
    frame=pd.DataFrame(rows)
    atomic_write(path,lambda f:f.write(frame.to_csv(index=False).encode('utf-8')),replace=Path(path).exists())


def append_csv(path,row):
    path=Path(path);old=path.read_bytes() if path.exists() else b''
    fields=next(csv.reader(io.StringIO(old.decode('utf-8')))) if old else list(row)
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n')
    if not old:w.writeheader()
    w.writerow(row)
    atomic_write(path,lambda f:f.write(old+buf.getvalue().encode('utf-8')),replace=path.exists())


@contextmanager
def single_runner(path):
    import msvcrt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0,2)
        if not f.tell():f.write(b'0');f.flush()
        f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as e:raise RuntimeError('B1恢复已在运行，不要重复启动') from e
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


class Checkpoints:
    """Committed local files plus the existing append-only SHA registry."""
    def __init__(self,local,journal):
        self.local,self.journal=Path(local),Path(journal)
        self.manifest={v['file']:v['sha256'] for v in read_jsonl(journal)}
    def register(self,path):
        key=Path(path).relative_to(self.local).as_posix()
        if key not in self.manifest:
            self.manifest[key]=b.b0.sha(path)
            append_jsonl(self.journal,{'file':key,'sha256':self.manifest[key]})
        return Path(path)
    def npz(self,path,compute):
        path=Path(path)
        if not path.exists():atomic_write(path,lambda f:np.savez_compressed(f,**compute()))
        self.register(path)
        return read_npz(path)
    def json(self,path,compute):
        path=Path(path)
        if not path.exists():write_json(path,compute())
        if path.is_relative_to(self.local):self.register(path)
        return read_json(path)


def commit_model(path,model,row):
    atomic_write(path,lambda f:pickle.dump({'model':model,'fit_receipt':row},f,protocol=pickle.HIGHEST_PROTOCOL))


def load_model(path):
    with Path(path).open('rb') as f:value=pickle.load(f)
    return value['model'] if isinstance(value,dict) and 'fit_receipt' in value else value


def recover_model(name,old_rows,path,train,register,append_fit):
    """Never call train for a fit with either a receipt or committed checkpoint."""
    if name in old_rows:
        row=old_rows[name]
        return load_model(row['local_model']),Path(row['local_model']),False
    path=Path(path)
    if path.exists():
        with path.open('rb') as f:value=pickle.load(f)
        if not isinstance(value,dict) or value['fit_receipt']['name']!=name:raise ValueError('Unknown checkpoint identity')
        model,row=value['model'],value['fit_receipt']
    else:
        model,row=train();commit_model(path,model,row)
    register(path)
    row={**row,'local_model':str(path),'sha256':b.b0.sha(path)}
    append_fit(row);old_rows[name]=row
    return model,path,True


def simple_checks():
    protected=read_json(ROOT/'artifacts/real_v2_b1_learning_curve_checks/protected_before.json')
    for key in ['configs/experiments/real_v2_b1_learning_curve.json','configs/data/real_v2_split.csv']:
        if b.b0.sha(ROOT/key)!=protected[key]['sha256']:raise ValueError('Frozen identity changed: '+key)
    expected=read_json(ROOT/'artifacts/real_v2_b0_diagnostics/execution_audit.json')['environment']['scikit-learn']
    if Path(sys.executable).resolve()!=Path('D:/python3.12.4/python.exe').resolve() or sys.version_info[:3]!=(3,12,4):raise ValueError('Interpreter changed')
    if importlib.metadata.version('scikit-learn')!=expected:raise ValueError('sklearn version changed')
    if not (sys.flags.ignore_environment and sys.flags.safe_path and sys.flags.utf8_mode and not sys.flags.no_user_site and sys.dont_write_bytecode):raise ValueError('Use -X utf8 -E -P -B')
    return {'status':'PASS','config_sha256':b.b0.sha(b.CFG),'split_sha256':b.b0.SPLIT_SHA,'python':sys.executable,
            'sklearn':expected,'old_122_models_23_OOF_not_rehashed':True,'no_raw_npz_identity_rescan':True}


def predict(model,x):
    if not len(x):return np.empty(0,float)
    p=np.concatenate([model.predict_proba(x[i:i+4096])[:,1] for i in range(0,len(x),4096)])
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid GBT probability')
    return p


def run():
    checks=simple_checks()
    if (OUT/'results.json').exists() and (OUT/'receipt.json').exists() and read_json(OUT/'receipt.json').get('status')=='COMPLETE':
        print('B1_ALREADY_COMPLETE_NO_ACTION',flush=True);return 0
    if not OUT.is_dir() or not LOCAL.is_dir():raise FileNotFoundError('Original B1 checkpoint directories required')
    with single_runner(LOCAL/'recovery.lock'):
        eligible,startup=b.b0.eligible_split(b.b0.SPLIT,b.b0.SPLIT_SHA)
        if len(eligible)!=315:raise ValueError('Development population changed')
        attempt=1
        while (OUT/f'resume_attempt_{attempt}.json').exists():attempt+=1
        write_json(OUT/f'resume_attempt_{attempt}.json',{'started':datetime.now(timezone.utc).isoformat(),'authority':'PI补充2','checks':checks})
        files=Checkpoints(LOCAL,OUT/'local_artifacts.jsonl')
        fit_rows=read_jsonl(OUT/'fit_receipts.jsonl');fits={r['name']:r for r in fit_rows}
        if len(fits)!=len(fit_rows):raise ValueError('Duplicate logical fit receipts')
        initial_fits=len(fits);new_times=[];eta_written=False
        accesses=b.boundary(eligible)
        raw_paths={str((DATA/k/(uid.replace('/','__')+'.npz')).resolve()) for uid in eligible.case_uid for k in ('windows','cases')}
        def access_log(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
                p=str(Path(os.fsdecode(args[0])).resolve())
                if p in raw_paths:append_jsonl(OUT/'resume_input_access.jsonl',{'attempt':attempt,'path':p,'time':datetime.now(timezone.utc).isoformat()})
        sys.addaudithook(access_log)
        started=time.perf_counter();stage='rehydrate_features'
        with (OUT/'run.log').open('a',encoding='utf-8',newline='\n') as journal:
            def note(*args):print(*args,flush=True);print(*args,file=journal,flush=True);os.fsync(journal.fileno())
            def output_json(path,compute):return files.json(path,compute)
            def append_fit(row):append_jsonl(OUT/'fit_receipts.jsonl',row)
            try:
                note('RESUME_START',attempt,'retained_fits',initial_fits,'PI补充2; numerical protocol unchanged')
                frame=b.b0.read_ledger(DATA/'ledgers/real_included_windows.csv',eligible)
                xx=[];ids=[]
                for uid in eligible.case_uid:
                    part=frame[frame.case_uid.eq(uid)];path=LOCAL/'window_features/real'/(uid.replace('/','__')+'.npz')
                    def make(part=part,uid=uid):
                        return {'features':b.b0.window_features(DATA/'windows'/(uid.replace('/','__')+'.npz'),part),
                                'original_cache_index':part.original_cache_index.to_numpy(),'labels':part.qualified_y.to_numpy()}
                    cached=files.npz(path,make)
                    if not np.array_equal(cached['original_cache_index'],part.original_cache_index) or not np.array_equal(cached['labels'],part.qualified_y):raise ValueError('Cached Real window identity differs')
                    xx.append(cached['features']);ids.extend(part.original_cache_index.tolist())
                x=np.concatenate(xx);del xx
                frame=frame.set_index('original_cache_index').loc[ids].reset_index();y=frame.qualified_y.to_numpy();masks=b.b0.populations(frame)
                with threadpool_limits(limits=16):
                    source_cache=LOCAL/'window_features/source_train.npz'
                    def make_source():
                        import torch
                        sf=pd.read_csv(DATA/'ledgers/train_included_windows.csv',keep_default_na=False)
                        cache=torch.load(DATA/'cache/train.pt',map_location='cpu',weights_only=True,mmap=True)
                        if cache['split']!='train' or cache['normalization_applied'] is not False or tuple(cache['x'].shape)!=(162069,26,150) or cache['x'].dtype!=torch.float32 or not np.array_equal(cache['y'].numpy(),sf.qualified_y) or not np.array_equal(cache['original_cache_index'].numpy(),sf.original_cache_index):raise ValueError('Source cache layout differs')
                        sx=np.concatenate([b.b0.stats(cache['x'][i:i+4096].numpy()) for i in range(0,len(cache['x']),4096)])
                        sy=cache['y'].numpy().copy();sw=b.b0.real_weights(sy,sf.normal_origin)
                        if not np.isfinite(sx).all():raise ValueError('Nonfinite source features')
                        return {'features':sx,'labels':sy,'weights':sw}
                    sd=files.npz(source_cache,make_source);sx,sy,sw=sd['features'],sd['labels'],sd['weights']
                    note('FEATURE_CACHES_READY','Real',len(eligible),'source',len(sy))
                    def fit(arm,mask,name,excluded):
                        nonlocal eta_written
                        if name in fits:return load_model(fits[name]['local_model']),Path(fits[name]['local_model'])
                        assert not frame.loc[mask,'dev_fold'].isin(excluded).any()
                        rx,ry=x[mask],y[mask];rw=b.b0.real_weights(ry,frame.loc[mask,'normal_origin'])
                        scale=1.
                        if arm=='S_plus_R':weights,scale=b.domain_weights(sw,rw);tx,ty=np.concatenate([sx,rx]),np.r_[sy,ry]
                        else:tx,ty,weights=rx,ry,rw
                        checkpoint=LOCAL/'models'/(name+'.checkpoint.pkl')
                        def train():
                            history=read_jsonl(OUT/'resume_fit_attempts.jsonl')
                            number=1+sum(v['name']==name for v in history)+(1 if name=='outer_S_plus_R_f0.50_rep1_k2' else 0)
                            append_jsonl(OUT/'resume_fit_attempts.jsonl',{'name':name,'physical_start_number':number,'resume_attempt':attempt,
                                'reason':'本机重启后的基础设施恢复' if number>1 else 'original remaining logical slot'})
                            note('RESUME_FIT_START',len(fits)+1,name,'physical_start',number)
                            t=time.perf_counter();model=b.HistGradientBoostingClassifier(**b.b0.RECIPE);model.fit(tx,ty,sample_weight=weights)
                            row={'name':name,'arm':arm,'real_windows':len(ry),'source_windows':len(tx)-len(ry),
                                'real_groups':int(frame.loc[mask,'association_group'].nunique()),'training_folds':sorted(map(int,frame.loc[mask,'dev_fold'].unique())),
                                'excluded_folds':list(excluded),'n_iter':int(model.n_iter_),'fit_time_s':time.perf_counter()-t,
                                'real_weight_scale':scale,'real_weight_total':float(rw.sum()*scale),'source_weight_total':float(sw.sum()) if arm=='S_plus_R' else 0.}
                            new_times.append(row['fit_time_s']);return model,row
                        model,path,_=recover_model(name,fits,checkpoint,train,files.register,append_fit)
                        note('RESUME_FIT_COMPLETE',len(fits),name,fits[name]['fit_time_s'])
                        if len(new_times)>=3 and not eta_written:
                            mean=float(np.mean(new_times[:3]));remaining=200-len(fits);fit_min=mean*remaining/60
                            eta={'based_on_first_new_fits':3,'mean_fit_s':mean,'remaining_fits':remaining,'fit_only_minutes':fit_min,
                                 'rough_total_minutes_range':[fit_min*2,fit_min*4],
                                 'caveat':'后续帧缓存/逐帧评分/11档事件聚合未实测，额外按拟合预算1至3倍预留；仅规划粗估，不是完成时限'}
                            write_json(OUT/f'resume_eta_{attempt}.json',eta);note('RESUME_ETA',json.dumps(eta,ensure_ascii=False));eta_written=True
                        return model,path
                    def window_pred(model,name,held):
                        p=files.npz(LOCAL/'window_predictions'/(name+'.npz'),lambda:{'probabilities':predict(model,x[held]),'original_cache_index':frame.loc[held,'original_cache_index'].to_numpy()})
                        if not np.array_equal(p['original_cache_index'],frame.loc[held,'original_cache_index']) or not np.isfinite(p['probabilities']).all():raise ValueError('Held prediction identity differs')
                        return p['probabilities']
                    # Restore lost gate probabilities from the five existing models only.
                    gate=read_json(OUT/'reproduction_gate.json')
                    if not gate['pass']:raise ValueError('Original reproduction gate did not pass')
                    gate_path=LOCAL/'oof/R_f1.00_rep0.npz'
                    gatep=None
                    if gate_path.exists():
                        files.register(gate_path);gatep=read_npz(gate_path)
                    if gatep is None:
                        p=np.full(len(y),np.nan)
                        for k in range(5):
                            name=f'outer_R_f1p00_rep0_k{k}';held=frame.dev_fold.eq(k).to_numpy()
                            if name not in fits:raise ValueError('Missing original gate model; do not refit gate')
                            model=load_model(fits[name]['local_model']);p[held]=window_pred(model,name,held);del model
                        gatep=files.npz(LOCAL/'oof/R_f1.00_rep0.npz',lambda:{'probabilities':p,'original_cache_index':np.asarray(ids,np.int64)})
                    note('GATE_MODELS_REUSED_NO_REFIT')
                    source_model=load_model(b.b0.SOURCE)
                    probabilities={('R',1.,0):gatep['probabilities']};full_models={('R',k):Path(fits[f'outer_R_f1p00_rep0_k{k}']['local_model']) for k in range(5)}
                    old_rows=[]
                    if (OUT/'window_metrics.csv').exists():
                        for row in csv.DictReader((OUT/'window_metrics.csv').open(encoding='utf-8',newline='')):
                            old_rows.append({k:(v if k=='arm' else int(v) if k=='repeat' else float(v) if v else None) for k,v in row.items()})
                    metrics={(r['arm'],r['fraction'],r['repeat']):r for r in old_rows}
                    window_rows=[];subsets=[]
                    def metric(arm,frac,rep,p):
                        key=(arm,frac,rep)
                        if key in metrics:return metrics[key]
                        def compute():
                            fpr=b.b0.fpr_at_recall(p,y,masks,.8)
                            return {'arm':arm,'fraction':frac,'repeat':rep,'AP':float(b.average_precision_score(y,p)),
                                'AUC_b':b.b0.weighted_auc(b.b0.auc_plan(p,y,(y==1)|masks['pre_fault']),np.ones(len(y),int)),
                                'threshold_recall08':fpr['threshold'],'actual_recall':fpr['actual_recall'],**{'FPR_'+n:v for n,v in fpr['normal_FPR'].items()}}
                        row=output_json(LOCAL/'state'/f'window_metric_{arm}_{frac:.2f}_{rep}.json',compute)
                        if arm!='S':append_csv(OUT/'window_metrics.csv',row)
                        metrics[key]=row;return row
                    stage='learning_curve'
                    for frac in b.FRACTIONS:
                        for rep in range(1 if frac==1 else 5):
                            for arm in b.ARMS:
                                key=(arm,frac,rep);path=LOCAL/'oof'/f'{arm}_f{frac:.2f}_rep{rep}.npz'
                                for k in range(5):
                                    if arm=='R':subsets.append({'outer_fold':k,'fraction':frac,'repeat':rep,'groups':b.subset_groups(eligible,k,frac,rep)})
                                if path.exists():
                                    cached=files.npz(path,lambda:None);p=cached['probabilities']
                                    if not np.array_equal(cached['original_cache_index'],ids):raise ValueError('OOF index identity differs')
                                    for k in range(5):
                                        held=frame.dev_fold.eq(k).to_numpy();name=f'outer_{arm}_f{frac:.2f}_rep{rep}_k{k}' if key!=('R',1.,0) else f'outer_R_f1p00_rep0_k{k}'
                                        files.npz(LOCAL/'window_predictions'/(name+'.npz'),lambda held=held:{'probabilities':p[held],'original_cache_index':frame.loc[held,'original_cache_index'].to_numpy()})
                                        if frac==1:full_models[(arm,k)]=Path(fits[name]['local_model'])
                                else:
                                    p=np.full(len(y),np.nan)
                                    for k in range(5):
                                        groups=b.subset_groups(eligible,k,frac,rep);tr=frame.association_group.isin(groups).to_numpy();held=frame.dev_fold.eq(k).to_numpy()
                                        assert not set(frame.loc[tr,'association_group'])&set(frame.loc[held,'association_group'])
                                        name=f'outer_{arm}_f{frac:.2f}_rep{rep}_k{k}'
                                        model,model_path=fit(arm,tr,name,[k]);p[held]=window_pred(model,name,held);del model
                                        if frac==1:full_models[(arm,k)]=model_path
                                    if not np.isfinite(p).all():raise ValueError('Incomplete OOF')
                                    files.npz(path,lambda:{'probabilities':p,'original_cache_index':np.asarray(ids,np.int64)})
                                probabilities[key]=p;window_rows.append(metric(arm,frac,rep,p))
                    sp=files.npz(LOCAL/'oof/S_f1.00_rep0.npz',lambda:{'probabilities':predict(source_model,x),'original_cache_index':np.asarray(ids,np.int64)})['probabilities']
                    source_window=metric('S',1.,0,sp)
                    output_json(OUT/'window_summary.json',lambda:{'rows':window_rows,'S_reference':source_window});output_json(OUT/'subsets.json',lambda:subsets)
                    stage='H1';h1=output_json(OUT/'H1.json',lambda:b.h1_bootstrap(probabilities,frame,eligible));note('H1_READY',h1['fractions']['0.25'])
                    stage='frame_features';from replay_ens_events_val import case_windows
                    cache_paths={}
                    for i,row in enumerate(eligible.itertuples(index=False),1):
                        uid=row.case_uid;key=uid.replace('/','__')+'.npz';path=LOCAL/'features'/key
                        def make_frame(key=key):
                            with np.load(DATA/'cases'/key,allow_pickle=False) as z:times,raw,valid,ref,lv=z['timestamp_us'],z['canonical_rows'],z['row_input_valid'],z['reference_state'],z['row_label_valid']
                            if times.dtype!=np.int64 or times.ndim!=1 or not len(times) or (np.diff(times)<=0).any() or raw.shape!=(len(times),26) or valid.dtype!=bool or valid.shape!=times.shape or lv.dtype!=bool or lv.shape!=times.shape or ref.shape!=times.shape or not np.isin(ref[lv],[0,1]).all():raise ValueError('Invalid Real frame layout')
                            ends,xr=case_windows(raw,valid) if len(raw)>=150 else (np.empty(0,np.int64),np.empty((0,26,150),np.float32))
                            ff=b.b0.stats(xr) if len(xr) else np.empty((0,130),np.float32)
                            if not np.isfinite(ff).all():raise ValueError('Nonfinite frame features')
                            return {'timestamp_us':times,'reference_state':ref,'row_label_valid':lv,'ends':ends,'features':ff}
                        files.npz(path,make_frame);cache_paths[uid]=path
                        if i%50==0:note('FRAME_CACHE_READY',i)
                    meta=eligible.set_index('case_uid')
                    def score_events(model,name,uids,thresholds):
                        rows=[]
                        for uid in uids:
                            key=uid.replace('/','__')+'.npz';cached_rows=LOCAL/'event_records'/name/(key+'.json')
                            def make_events(uid=uid,key=key):
                                arrays=read_npz(cache_paths[uid])
                                def make_score():
                                    p=np.full(len(arrays['timestamp_us']),np.nan,float);p[arrays['ends']]=predict(model,arrays['features']);return {'probabilities':p}
                                p=files.npz(LOCAL/'scores'/name/key,make_score)['probabilities']
                                return [b.event_record({'case_uid':uid,**meta.loc[uid].to_dict()},arrays,p,th) for th in thresholds]
                            values=output_json(cached_rows,make_events)
                            if [v['onset_th'] for v in values]!=list(thresholds):raise ValueError('Cached event onsets differ; never change outer selection')
                            rows.extend(values)
                        return rows
                    rows_by_arm={a:[] for a in (*b.ARMS,'S')};selections=[];scans=[]
                    stage='nested_events'
                    for k in range(5):
                        for arm in b.ARMS:
                            def select_inner(k=k,arm=arm):
                                inner=[]
                                for j in range(5):
                                    if j==k:continue
                                    mask=~frame.dev_fold.isin([k,j]).to_numpy();name=f'inner_{arm}_outer{k}_held{j}'
                                    model,_=fit(arm,mask,name,[k,j]);inner.extend(score_events(model,name,eligible.loc[eligible.dev_fold.eq(j),'case_uid'],b.GRID));del model
                                path=LOCAL/'inner_events'/f'{arm}_outer{k}.csv'
                                if not path.exists():write_csv(path,inner);files.register(path)
                                grid=[{'onset_th':th,**b.event_summary([r for r in inner if r['onset_th']==th])} for th in b.GRID]
                                choice=b.select_onset(grid);choice.update(arm=arm,outer_fold=k)
                                return {'choice':choice,'grid':grid}
                            selected=output_json(LOCAL/'state'/f'selection_{arm}_{k}.json',select_inner)
                            choice=selected['choice'];selections.append(choice);scans.append({'arm':arm,'outer_fold':k,'grid':selected['grid']})
                            selection_rows=[{'arm':v['arm'],'outer_fold':v['outer_fold'],'onset_th':v['onset_th'],'no_feasible_onset':v['no_feasible_onset'],
                                'tied_onsets_primary':json.dumps(v['tied_onsets_primary']),'tied_onsets_after_detection':json.dumps(v['tied_onsets_after_detection']),
                                'inner_N1_rate':v['inner']['N1']['rate'],'inner_N2_rate':v['inner']['N2']['rate'],
                                'inner_motor':v['inner']['motor']['detection'],'inner_sensor':v['inner']['sensor']['detection']} for v in selections]
                            write_csv(OUT/'onset_selection.csv',selection_rows)
                            path=full_models[(arm,k)];model=load_model(path)
                            # Stable logical names work for both old plain models and new checkpoints.
                            name=next(n for n,v in fits.items() if Path(v['local_model'])==path)
                            rows_by_arm[arm].extend(score_events(model,name,eligible.loc[eligible.dev_fold.eq(k),'case_uid'],[choice['onset_th']]));del model
                            note('OUTER_EVENT_READY',arm,k,choice['onset_th'],choice['no_feasible_onset'])
                    rows_by_arm['S']=score_events(source_model,'frozen_S',eligible.case_uid,[.95])
                    for arm,rows in rows_by_arm.items():
                        for row in rows:row['arm']=arm
                    write_csv(OUT/'events_per_case.csv',[v for rows in rows_by_arm.values() for v in rows])
                    output_json(OUT/'inner_selection.json',lambda:{'selections':selections,'scans':scans})
                    event_point=output_json(OUT/'event_points.json',lambda:{a:b.event_summary(rows) for a,rows in rows_by_arm.items()})
                    stage='event_CI';event_ci=output_json(OUT/'event_CI.json',lambda:b.event_bootstrap(rows_by_arm,eligible))
                    def make_paired():
                        rows=[]
                        for arm in b.ARMS:
                            p=probabilities[(arm,1.,0)]
                            for uid,part in frame[frame.family.isin(['motor','sensor'])].groupby('case_uid',sort=True):
                                same=frame.case_uid.eq(uid).to_numpy();mask=same&((y==1)|masks['pre_fault'])
                                rows.append({'arm':arm,'case_uid':uid,'family':part.family.iloc[0],
                                    'AUC':b.b0.weighted_auc(b.b0.auc_plan(p,y,mask),np.ones(len(y),int)),
                                    'fault_windows':int((same&(y==1)).sum()),'pre_fault_windows':int((same&masks['pre_fault']).sum())})
                        summary={}
                        for arm in b.ARMS:
                            summary[arm]={}
                            for fam in ('motor','sensor'):
                                values=[v['AUC'] for v in rows if v['arm']==arm and v['family']==fam and v['AUC'] is not None];n=sum(v['arm']==arm and v['family']==fam for v in rows)
                                summary[arm][fam]={'cases':n,'defined':len(values),'undefined':n-len(values),'median':float(np.median(values)) if values else None,
                                    'q25':float(np.quantile(values,.25)) if values else None,'q75':float(np.quantile(values,.75)) if values else None,
                                    'min':float(min(values)) if values else None,'max':float(max(values)) if values else None}
                        return {'rows':rows,'summary':summary}
                    paired=output_json(LOCAL/'state/paired_case_AUC.json',make_paired);write_csv(OUT/'paired_case_AUC.csv',paired['rows'])
                    verdicts={a:b.h2(event_point[a]) for a in b.ARMS};any_h2=any(v['holds'] for v in verdicts.values())
                    recorded=read_jsonl(OUT/'resume_input_access.jsonl');paths={v['path'] for v in recorded};assert paths<=raw_paths
                    access={'recorded_since_supplement2':recorded,'unique_windows':len([p for p in paths if Path(p).parent.name=='windows']),
                            'unique_cases':len([p for p in paths if Path(p).parent.name=='cases']),'raw_open_calls':len(recorded),'dev_only':True,
                            'initial_attempt_access_counter_lost_in_RAM':'Code path loaded315 dev windows; no raw cases before interruption. Not added as measured open calls.'}
                    assert len(fits)==200 and len(selections)==10
                    result=output_json(OUT/'results.json',lambda:{'status':'COMPLETE','protocol':str(b.CFG.relative_to(ROOT)),'config_sha256':checks['config_sha256'],
                        'startup':startup,'gate':gate,'recipe':b.b0.RECIPE,'fractions':b.FRACTIONS,'window_metrics':window_rows,'S_window_reference':source_window,
                        'H1':h1,'onset_selection':selections,'event':event_point,'event_CI':event_ci,'H2':{'arms':verdicts,'line_holds':any_h2},
                        'descriptive':{'by_flight':{a:{f:b.event_summary([v for v in rows if v['flight']==f]) for f in sorted(eligible.flight.unique())} for a,rows in rows_by_arm.items()},'paired_case_AUC':paired['summary']},
                        'consequences':b.consequences(h1['fractions']['0.25']['holds'],any_h2),'fit_count':len(fits),'runtime_root':str(LOCAL),'access':access,
                        'resume_attempt':attempt,'retained_model_count_first_resume':122,'physical_restart_slot123':'本机重启后的基础设施恢复，不是重跑指标',
                        'wall_time_this_resume_s':time.perf_counter()-started,'no_sealed_internal_test_or_GPU':True,'no_automatic_B2':True})
                    output_json(OUT/'receipt.json',lambda:{'status':'COMPLETE','config_sha256':checks['config_sha256'],'split_sha256':b.b0.SPLIT_SHA,
                        'S_model_sha256':b.b0.SOURCE_SHA,'runtime_root':str(LOCAL),'local_artifacts_sha256':files.manifest,'fit_count':len(fits),'fits':list(fits.values()),
                        'selections':selections,'access':access,'checks':checks,'results_sha256':b.b0.sha(OUT/'results.json'),'resume_attempt':attempt,
                        'old_checkpoint_files_not_rehashed':True,'no_source_or_real_raw_identity_rescan':True,'io_events_this_attempt':IO_EVENTS})
                    note('COMPLETE',json.dumps(result['H2']),result['consequences']['key']);return 0
            except Exception as exc:
                note('RESUME_FAILED_STOP',stage,repr(exc))
                write_json(OUT/f'resume_failure_{attempt}.json',{'stage':stage,'error':repr(exc),'fits':len(fits),'attempt':attempt,'io_events':IO_EVENTS})
                raise


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(run())
