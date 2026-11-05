"""Independent reconciliation of c-only support; inputs v1/v2 CSVs only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
SRC=OUT.parent/'M4_design_preflight_v1_20261105'
FEATURES=['vz_median_mps','speed_median_mps','speed_std_mps','gyro_p90_radps']


def read(p):return pd.read_csv(p,dtype={'association_group':str},keep_default_na=False)


def weight(f):
    g=f[['case_uid','association_group']].drop_duplicates().groupby('association_group').size()
    n=f.groupby('case_uid').original_cache_index.transform('size')
    return 1/(f.association_group.nunique()*f.association_group.map(g).to_numpy()*n.to_numpy())


def quant(x,w,ps):
    order=np.argsort(x,kind='stable');cs=np.cumsum(w[order]);cs/=cs[-1]
    return np.asarray(x)[order][np.minimum(np.searchsorted(cs,ps,side='left'),len(cs)-1)]


def main():
    f=read(SRC/'window_state_ledger.csv.gz');nested=read(SRC/'nested_group_assignments_3.csv')
    normal={'D1':f.known_normal,'D2':f.known_normal & f.normal_origin.eq('normal_flight'),'D3':f.known_normal & f.normal_origin.eq('normal_part_of_fault_case')}
    assert np.array_equal(normal['D1'],normal['D2']|normal['D3'])
    for col in ['original_cache_index','case_uid','association_group']:
        assert not set(f.loc[normal['D2'],col])&set(f.loc[normal['D3'],col])
    saved_scalers=pd.read_csv(OUT/'condition_scalers.csv')
    threshold=pd.read_csv(OUT/'distance_thresholds_fit_normal_LOO.csv')
    ledger=read(OUT/'query_identity.csv.gz')
    pd.testing.assert_frame_equal(f,ledger,check_dtype=False,atol=1e-12,rtol=1e-12)
    probes=[];err={'ESS':0.,'distance':0.,'mass':0.}; cross=[]; normaldiag=[]
    gate='supported_G15_C30_D10_Q99'
    for fold in range(3):
        roles=f.association_group.map(nested[nested.outer_evaluation_fold.eq(fold)].set_index('association_group').role)
        assert roles.notna().all()
        assert pd.DataFrame({'case':f.case_uid,'role':roles}).groupby('case').role.nunique().eq(1).all()
        for pop,mask in normal.items():
            rr=read(OUT/f'condition_support_queries_f{fold}_{pop}.csv.gz')
            assert len(rr)==162069 and rr.original_cache_index.is_unique
            for col in ['original_cache_index','case_uid','association_group']:
                assert np.array_equal(rr[col],f[col]),col
            assert np.array_equal(rr.role,roles)
            ref=f[mask & roles.eq('inner_fit')].copy();aw=weight(ref)
            x=ref[FEATURES].to_numpy();mu=aw@x;sd=np.maximum(np.sqrt(aw@((x-mu)**2)),1e-8)
            ss=saved_scalers[(saved_scalers.outer_fold==fold)&(saved_scalers.population==pop)].set_index('feature').loc[FEATURES]
            np.testing.assert_allclose(mu,ss['mean'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(sd,ss.std_used,rtol=1e-12,atol=1e-12)
            cm=ref.groupby(['case_uid','association_group','domain'],sort=True)[FEATURES].mean().reset_index()
            gm=cm.groupby('association_group',sort=True)[FEATURES].mean()
            rgroup=ref.association_group.to_numpy(); rcase=ref.case_uid.to_numpy();domain=ref.domain.to_numpy()
            picks=set()
            for role in ['inner_fit','inner_dev','outer_held_out']:
                for label in [0,1]:
                    idx=np.flatnonzero(roles.eq(role).to_numpy()&f.qualified_y.eq(label).to_numpy())
                    picks.update([int(idx[0]),int(idx[len(idx)//2]),int(idx[-1])])
            picks.update([int(rr.group_ESS.idxmin()),int(rr.nearest_group_distance.idxmax())])
            for ix in sorted(picks):
                query=f.iloc[ix];keep=np.ones(len(ref),dtype=bool)
                if roles.iloc[ix]=='inner_fit':keep=rgroup!=query.association_group
                dif=(x[keep]-query[FEATURES].to_numpy(dtype=float))/sd
                logw=np.log(aw[keep])-np.einsum('ij,ij->i',dif,dif)/8
                w=np.exp(logw-logw.max());w/=w.sum()
                cw=pd.Series(w).groupby(rcase[keep]).sum()
                gw=pd.Series(w).groupby(rgroup[keep]).sum()
                dw={}
                for d in ['SIL','HIL']:
                    cweights=pd.Series(w[domain[keep]==d]).groupby(rcase[keep][domain[keep]==d]).sum().to_numpy()
                    mass=cweights.sum();dw[d+'_mass']=mass
                    dw[d+'_case_ESS']=mass*mass/(cweights@cweights) if cweights@cweights>0 else 0
                metrics={'group_ESS':1/(gw.to_numpy()@gw.to_numpy()),'case_ESS':1/(cw.to_numpy()@cw.to_numpy()),**dw}
                casekeep=cm.association_group.ne(query.association_group) if roles.iloc[ix]=='inner_fit' else np.ones(len(cm),bool)
                groupkeep=gm.index!=query.association_group if roles.iloc[ix]=='inner_fit' else np.ones(len(gm),bool)
                dc=np.sqrt((((cm.loc[casekeep,FEATURES].to_numpy()-query[FEATURES].to_numpy(dtype=float))/sd)**2).sum(1))
                dg=np.sqrt((((gm.loc[groupkeep,FEATURES].to_numpy()-query[FEATURES].to_numpy(dtype=float))/sd)**2).sum(1))
                metrics.update(nearest_case_distance=dc.min(),nearest_group_distance=dg.min(),mean5_case_distance=np.sort(dc)[:5].mean())
                for name,val in metrics.items():
                    saved=float(rr.loc[ix,name]);error=abs(val-saved)
                    category='distance' if 'distance' in name else ('mass' if 'mass' in name else 'ESS')
                    err[category]=max(err[category],error)
                    np.testing.assert_allclose(val,saved,rtol=2e-8,atol=1e-6 if category=='distance' else 1e-8)
                probes.append({'outer_fold':fold,'population':pop,'original_cache_index':int(query.original_cache_index),'case_uid':query.case_uid,'association_group':query.association_group,'role':roles.iloc[ix],'whole_group_excluded':roles.iloc[ix]=='inner_fit','max_absolute_error':max(abs(v-float(rr.loc[ix,k])) for k,v in metrics.items())})
            np.testing.assert_allclose(rr.SIL_mass+rr.HIL_mass,1,atol=1e-12,rtol=1e-12)
            fitidx=np.flatnonzero(mask.to_numpy()&roles.eq('inner_fit').to_numpy());ww=weight(f.iloc[fitidx])
            ts={}
            for distance in ['nearest_case_distance','nearest_group_distance','mean5_case_distance']:
                tq=quant(rr.loc[fitidx,distance].to_numpy(),ww,[.95,.99])
                for level,t in zip([95,99],tq):
                    saved=threshold[(threshold.outer_fold==fold)&(threshold.population==pop)&(threshold.distance==distance)&(threshold.percentile==level)].threshold.iloc[0]
                    np.testing.assert_allclose(t,saved,atol=1e-12,rtol=1e-12)
                    ts[(distance,level)]=saved
            for g,c,d in [(8,16,5),(15,30,10),(30,60,20)]:
                es=rr.group_ESS.ge(g)&rr.case_ESS.ge(c)&rr.SIL_case_ESS.ge(d)&rr.HIL_case_ESS.ge(d)&rr.SIL_mass.ge(.05)&rr.HIL_mass.ge(.05)
                for level in [95,99]:
                    expected=es & rr.nearest_group_distance.le(ts[('nearest_group_distance',level)])&f.state.ne('descent_proxy')
                    assert np.array_equal(expected,rr[f'supported_G{g}_C{c}_D{d}_Q{level}'])
            # Additional crossed fault/domain coverage, retaining fixed master IDs.
            for fault in ['GPS','accelerometer','barometer','gyroscope','magnetometer','motor']:
                for dom in ['SIL','HIL']:
                    ix=np.flatnonzero(roles.eq('outer_held_out') & f.qualified_y.eq(1)&f.fault_directory.eq(fault)&f.domain.eq(dom))
                    ok=rr.loc[ix,gate].to_numpy(dtype=bool)
                    cross.append({'outer_fold':fold,'population':pop,'fault_directory':fault,'domain':dom,'queries':len(ix),'cases':f.iloc[ix].case_uid.nunique(),'groups':f.iloc[ix].association_group.nunique(),'supported':int(ok.sum()),'fallback':int((~ok).sum()),'coverage':ok.mean()})
            for origin in ['normal_flight','normal_part_of_fault_case']:
                ix=np.flatnonzero(roles.eq('outer_held_out') & f.normal_origin.eq(origin))
                normaldiag.append({'outer_fold':fold,'population':pop,'origin':origin,'queries':len(ix),'supported':int(rr.loc[ix,gate].sum()),'coverage':rr.loc[ix,gate].mean()})
    # AUC and pair data are copied descriptive evidence, not new model scores.
    old=pd.read_csv(SRC/'conditional_case_AUC_descriptive.csv');old=old[old.feature.isin(FEATURES)].reset_index(drop=True)
    new=pd.read_csv(OUT/'condition_AUC_all_fixed_direction.csv')
    pd.testing.assert_frame_equal(old,new,atol=1e-12,rtol=1e-12)
    pairs=read(SRC/'case_pre_post_state_pairs.csv');pd.testing.assert_frame_equal(pairs,read(OUT/'pre_post_identity_changes.csv'))
    for name,data in [('independent_query_probes.csv',pd.DataFrame(probes)),('heldout_fault_domain_coverage.csv',pd.DataFrame(cross)),('heldout_normal_origin_coverage.csv',pd.DataFrame(normaldiag))]:data.to_csv(OUT/name,index=False,mode='x')
    record={'status':'PASS_V2_C_ONLY_INDEPENDENT_RECONCILIATION','all9_query_tables_IDs_and_roles_checked':True,'total_query_rows':9*162069,'direct_independent_queries':len(probes),'max_absolute_errors':err,'all6_gate_flags_and_all_normal_LOO_distance_thresholds_checked':True,'group_isolation':True,'population_ID_case_group_disjoint_union':True,'v1_AUC_and_prepost_unchanged':True,'no_models_h_U_C_scores_or_source_data_reads':True,'limits':'Direct numerical checks are sampled; all IDs/gate flags/thresholds checked. This verifies c support only, not normal h support, diagnostic performance or fault causality.'}
    with (OUT/'independent_verification.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
