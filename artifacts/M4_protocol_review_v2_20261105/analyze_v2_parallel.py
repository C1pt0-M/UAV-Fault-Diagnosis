"""Published-v1 CSV-only CPU audit. No ML/model imports or source data reads."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import hashlib
import io
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SRC = OUT.parent / 'M4_design_preflight_v1_20261105'
FEATURES = ['vz_median_mps', 'speed_median_mps', 'speed_std_mps', 'gyro_p90_radps']
STATES = ['climb_proxy', 'low_motion_proxy', 'maneuver_proxy', 'translation_proxy', 'other_motion', 'descent_proxy']


def csv(name, f):
    target=OUT/name
    if target.exists():
        old=pd.read_csv(target,dtype={'association_group':str},keep_default_na=False)
        expected=pd.read_csv(io.StringIO(f.to_csv(index=False)),dtype={'association_group':str},keep_default_na=False)
        pd.testing.assert_frame_equal(expected,old,check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)
    else:
        f.to_csv(target,index=False,mode='x',compression='infer')


def js(name, x):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)


def weights(f):
    n = f.groupby('case_uid').original_cache_index.transform('size').to_numpy()
    cases = f[['case_uid', 'association_group']].drop_duplicates()
    nc = cases.groupby('association_group').size()
    return 1 / (f.association_group.nunique() * f.association_group.map(nc).to_numpy() * n)


def wquantile(x, w, ps):
    order = np.argsort(x, kind='stable')
    xx, ww = np.asarray(x)[order], np.asarray(w)[order]
    cumulative = np.cumsum(ww) / ww.sum()
    return xx[np.minimum(np.searchsorted(cumulative, ps, side='left'), len(xx)-1)]


def count(f):
    sizes = f[['case_uid', 'association_group']].drop_duplicates().groupby('association_group').size()
    return {'windows': len(f), 'cases': f.case_uid.nunique(), 'groups': f.association_group.nunique(),
            'SIL_cases': f[f.domain.eq('SIL')].case_uid.nunique(), 'HIL_cases': f[f.domain.eq('HIL')].case_uid.nunique(),
            'paired_groups': int(sizes.eq(2).sum()), 'singleton_groups': int(sizes.eq(1).sum())}


def normal_counts(f, base):
    rows = []
    for typ, column, levels in [('all', None, ['all']), ('state', 'state', STATES),
                                 ('flight', 'flight_directory', ['acce','dece','hover','velocity','waypoint','circling']),
                                 ('domain', 'domain', ['SIL','HIL'])]:
        for level in levels:
            p = f if column is None else f[f[column].eq(level)]
            c = count(p)
            rows.append({**base, 'stratum': typ, 'level': level, **c,
                         'v1_reference_count_gate': c['windows']>=1000 and c['cases']>=30 and c['groups']>=15 and c['SIL_cases']>=10 and c['HIL_cases']>=10})
    return rows


def distances(q, representatives):
    return np.sqrt(np.maximum((q*q).sum(1)[:,None] + (representatives*representatives).sum(1)[None,:] - 2*q@representatives.T, 0))


def main():
    started=time.perf_counter()
    plan=json.loads((OUT/'analysis_plan.json').read_text(encoding='utf-8'))
    f=pd.read_csv(SRC/'window_state_ledger.csv.gz',dtype={'association_group':str},keep_default_na=False)
    assert len(f)==162069 and f.original_cache_index.is_unique
    assert f.case_uid.nunique()==2570 and f.association_group.nunique()==1296
    group_codes={g:i for i,g in enumerate(sorted(f.association_group.unique()))}
    definitions={'D1':f.known_normal.to_numpy(), 'D2':(f.known_normal & f.normal_origin.eq('normal_flight')).to_numpy(),
                 'D3':(f.known_normal & f.normal_origin.eq('normal_part_of_fault_case')).to_numpy()}
    assert not (definitions['D2'] & definitions['D3']).any()
    assert np.array_equal(definitions['D1'],definitions['D2']|definitions['D3'])
    assert [int(definitions[x].sum()) for x in definitions]==[83051,21145,61906]
    a=pd.read_csv(SRC/'nested_group_assignments_3.csv',dtype={'association_group':str})
    for outer in range(3):
        aa=a[a.outer_evaluation_fold.eq(outer)]
        roles=f.association_group.map(aa.set_index('association_group').role)
        assert roles.notna().all() and pd.DataFrame({'case':f.case_uid,'role':roles}).groupby('case').role.nunique().eq(1).all()
    csv('query_identity.csv.gz',f)
    support=[]; distribution=[]; mass=[]; group_contributions=[]
    for population,mask in definitions.items():
        p=f[mask]; support+=normal_counts(p,{'population':population,'outer_fold':-1,'role':'all_Train_description'})
        w=weights(p)
        for (origin,phase), indexes in p.groupby(['normal_origin','phase']).groups.items():
            positions=p.index.get_indexer(indexes)
            mass.append({'population':population,'normal_origin':origin,'phase':phase,'windows':len(indexes),'group_equal_mass':float(w[positions].sum())})
        g=p[['case_uid','association_group','domain','flight_directory','normal_origin']].drop_duplicates()
        g['windows']=g.case_uid.map(p.groupby('case_uid').size())
        g['population']=population
        g['group_mass']=1/p.association_group.nunique()
        g['case_mass']=g.group_mass/g.association_group.map(g.groupby('association_group').size())
        group_contributions.append(g)
        for outer in range(3):
            role=f.association_group.map(a[a.outer_evaluation_fold.eq(outer)].set_index('association_group').role)
            for r in ['inner_fit','inner_dev','outer_held_out']:
                part=f[mask & role.eq(r).to_numpy()]
                base={'population':population,'outer_fold':outer,'role':r}
                support+=normal_counts(part,base)
                for typ,col in [('all',None),('state','state'),('domain','domain'),('flight','flight_directory')]:
                    strata=[('all',part)] if col is None else part.groupby(col,sort=True)
                    for level,pp in strata:
                        ww=weights(pp)
                        for feature in FEATURES:
                            x=pp[feature].to_numpy()
                            quant=wquantile(x,ww,[.1,.25,.5,.75,.9,.95,.99])
                            distribution.append({**base,'stratum':typ,'level':level,'feature':feature,'windows':len(pp),'cases':pp.case_uid.nunique(),'groups':pp.association_group.nunique(),
                                                 'weighted_mean':float(ww@x),'weighted_std':float(np.sqrt(ww@((x-ww@x)**2))),
                                                 **dict(zip(['p10','p25','p50','p75','p90','p95','p99'],quant))})
    csv('normal_population_support.csv',pd.DataFrame(support))
    csv('normal_population_distributions.csv',pd.DataFrame(distribution))
    csv('normal_origin_phase_mass.csv',pd.DataFrame(mass))
    csv('normal_group_case_contributions.csv',pd.concat(group_contributions))
    auc=pd.read_csv(SRC/'conditional_case_AUC_descriptive.csv')
    auc=auc[auc.feature.isin(FEATURES)].copy()
    csv('condition_AUC_all_fixed_direction.csv',auc)
    # Preserve underlying case identities for every AUC stratum, no new scores.
    case=pd.read_csv(SRC/'case_state_feature_summaries.csv.gz',dtype={'association_group':str},keep_default_na=False)
    cols=['case_uid','association_group','domain','fault_directory','flight_directory','state','known_label']+FEATURES
    csv('condition_case_identity_values.csv.gz',case[cols])
    pairs=pd.read_csv(SRC/'case_pre_post_state_pairs.csv',dtype={'association_group':str},keep_default_na=False)
    csv('pre_post_identity_changes.csv',pairs)
    rows=[]
    for (fault,domain,changed),p in pairs.groupby(['fault_directory','domain','state_changed']):
        for feature in FEATURES:
            x=p[feature+'_delta'].to_numpy()
            rows.append({'fault_directory':fault,'domain':domain,'state_changed':changed,'feature':feature,'cases':len(p),'groups':p.association_group.nunique(),
                         'negative_cases':int((x<0).sum()),'zero_cases':int((x==0).sum()),'positive_cases':int((x>0).sum()),
                         **dict(zip(['p10','p25','p50','p75','p90'],np.quantile(x,[.1,.25,.5,.75,.9]))),
                         'median_gap_s':p.endpoint_gap_s.median()})
    csv('pre_post_delta_by_fault_domain_route.csv',pd.DataFrame(rows))
    all_summaries=[]; coverage=[]; scales=[]; thresholds=[]; checks=[]
    for outer in range(3):
        role=f.association_group.map(a[a.outer_evaluation_fold.eq(outer)].set_index('association_group').role).to_numpy()
        for population,mask in definitions.items():
            print('START',outer,population,round(time.perf_counter()-started,2),flush=True)
            ref=f[mask & (role=='inner_fit')].sort_values(['association_group','case_uid','original_cache_index']).copy()
            rw=weights(ref); values=ref[FEATURES].to_numpy(dtype=np.float64)
            mean=rw@values; var=rw@((values-mean)**2); rawsd=np.sqrt(var); sd=np.maximum(rawsd,1e-8)
            z=(values-mean)/sd; q=(f[FEATURES].to_numpy(dtype=np.float64)-mean)/sd
            case_ids=ref.case_uid.to_numpy(); cstarts=np.r_[0,np.flatnonzero(case_ids[1:]!=case_ids[:-1])+1]
            group_ids=ref.association_group.map(group_codes).to_numpy(); cg=group_ids[cstarts]
            gstarts=np.r_[0,np.flatnonzero(cg[1:]!=cg[:-1])+1]; gu=cg[gstarts]
            domains=ref.domain.to_numpy()[cstarts]; sil=domains=='SIL'
            nper=np.diff(np.r_[cstarts,len(ref)])
            centers=np.add.reduceat(values,cstarts,axis=0)/nper[:,None]
            groupcenters=np.add.reduceat(centers,gstarts,axis=0)/np.diff(np.r_[gstarts,len(centers)])[:,None]
            ccenters=(centers-mean)/sd; gcenters=(groupcenters-mean)/sd
            for j,name in enumerate(FEATURES): scales.append({'outer_fold':outer,'population':population,'feature':name,'mean':mean[j],'std_raw':rawsd[j],'std_used':sd[j],'floor':1e-8,'zero_variance':bool(var[j]==0),'fit_windows':len(ref),'fit_cases':len(cstarts),'fit_groups':len(gu)})
            ctable=ref.iloc[cstarts][['case_uid','association_group','domain']].copy()
            for j,name in enumerate(FEATURES):ctable[name+'_mean']=centers[:,j]
            csv(f'case_representatives_f{outer}_{population}.csv',ctable)
            gtable=pd.DataFrame({'association_group':ref.association_group.to_numpy()[cstarts][gstarts]})
            for j,name in enumerate(FEATURES):gtable[name+'_mean']=groupcenters[:,j]
            csv(f'group_representatives_f{outer}_{population}.csv',gtable)
            result=np.empty((len(f),11),dtype=np.float64)
            groups_query=f.association_group.map(group_codes).to_numpy(); loga=np.log(rw)
            def process_batch(begin):
                end=min(begin+256,len(f)); qq=q[begin:end]
                dist2=np.maximum((qq*qq).sum(1)[:,None]+(z*z).sum(1)[None,:]-2*qq@z.T,0)
                logs=-dist2/8+loga[None,:]
                own=(groups_query[begin:end,None]==group_ids[None,:]) & (role[begin:end,None]=='inner_fit')
                logs[own]=-np.inf
                logs-=np.max(logs,axis=1)[:,None]
                np.exp(logs,out=logs)
                cw=np.add.reduceat(logs,cstarts,axis=1)
                cw/=cw.sum(1)[:,None]
                gw=np.add.reduceat(cw,gstarts,axis=1)
                ms=cw[:,sil].sum(1); mh=cw[:,~sil].sum(1)
                es=np.divide(ms*ms,(cw[:,sil]**2).sum(1),out=np.zeros(len(qq)),where=(cw[:,sil]**2).sum(1)>0)
                eh=np.divide(mh*mh,(cw[:,~sil]**2).sum(1),out=np.zeros(len(qq)),where=(cw[:,~sil]**2).sum(1)>0)
                dc=distances(qq,ccenters); dg=distances(qq,gcenters)
                dc[(groups_query[begin:end,None]==cg[None,:]) & (role[begin:end,None]=='inner_fit')]=np.inf
                dg[(groups_query[begin:end,None]==gu[None,:]) & (role[begin:end,None]=='inner_fit')]=np.inf
                result[begin:end]=np.column_stack([1/(gw*gw).sum(1),1/(cw*cw).sum(1),ms,mh,es,eh,np.minimum(es,eh),dc.min(1),dg.min(1),np.partition(dc,4,axis=1)[:,:5].mean(1),cw.max(1)])
                if begin%(256*120)==0:print('PROGRESS',outer,population,end,len(f),round(time.perf_counter()-started,1),flush=True)
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(process_batch,range(0,len(f),256)))
            names=['group_ESS','case_ESS','SIL_mass','HIL_mass','SIL_case_ESS','HIL_case_ESS','min_domain_case_ESS','nearest_case_distance','nearest_group_distance','mean5_case_distance','max_case_weight']
            assert np.isfinite(result).all()
            rr=pd.DataFrame(result,columns=names)
            for key in ['original_cache_index','case_uid','association_group']:rr[key]=f[key].to_numpy()
            rr['role']=role;rr['outer_fold']=outer;rr['population']=population
            fitidx=np.flatnonzero(mask & (role=='inner_fit'))
            fitweights=weights(f.iloc[fitidx])
            for distance in names[7:10]:
                tq=wquantile(rr.loc[fitidx,distance].to_numpy(),fitweights,[.95,.99])
                for percentile,t in zip([95,99],tq):thresholds.append({'outer_fold':outer,'population':population,'distance':distance,'percentile':percentile,'threshold':t,'fit_queries':len(fitidx),'fit_query_policy':'whole-group excluded'})
            td=wquantile(rr.loc[fitidx,'nearest_group_distance'].to_numpy(),fitweights,[.95,.99])
            for gi,ci,di in plan['ESS_candidates']:
                ess=(rr.group_ESS>=gi)&(rr.case_ESS>=ci)&(rr.SIL_case_ESS>=di)&(rr.HIL_case_ESS>=di)
                ess &= (rr.SIL_mass>=.05)&(rr.HIL_mass>=.05)
                for percentile,t in zip([95,99],td):
                    key=f'G{gi}_C{ci}_D{di}_Q{percentile}'
                    joint=ess & rr.nearest_group_distance.le(t) & f.state.ne('descent_proxy')
                    rr['supported_'+key]=joint.to_numpy()
            rr['ESS_only_primary']=(rr.group_ESS.ge(15)&rr.case_ESS.ge(30)&rr.SIL_case_ESS.ge(10)&rr.HIL_case_ESS.ge(10)&rr.SIL_mass.ge(.05)&rr.HIL_mass.ge(.05))
            rr['distance_only_Q99']=rr.nearest_group_distance.le(td[1])
            csv(f'condition_support_queries_f{outer}_{population}.csv.gz',rr)
            for r in ['inner_fit','inner_dev','outer_held_out']:
                rolesel=role==r
                strata=[('all','all',np.ones(len(f),dtype=bool)),('label','normal',f.qualified_y.eq(0).to_numpy()),('label','fault',f.qualified_y.eq(1).to_numpy())]
                for col in ['domain','state','flight_directory','fault_directory','normal_origin']:
                    strata += [(col,key,f[col].eq(key).to_numpy()) for key in sorted(f[col].unique())]
                for typ,level,sel in strata:
                    ix=np.flatnonzero(rolesel & sel)
                    if not len(ix):continue
                    pop=f.iloc[ix]; weights_query=weights(pop)
                    for name in names:
                        x=rr.loc[ix,name].to_numpy()
                        all_summaries.append({'outer_fold':outer,'population':population,'role':r,'stratum':typ,'level':level,'metric':name,'queries':len(ix),'cases':pop.case_uid.nunique(),'groups':pop.association_group.nunique(),'minimum':x.min(),
                                              **dict(zip(['p10','p25','p50','p75','p90','p95','p99'],np.quantile(x,[.1,.25,.5,.75,.9,.95,.99])))})
                    for name in [n for n in rr.columns if n.startswith('supported_')]+['ESS_only_primary','distance_only_Q99']:
                        ok=rr.loc[ix,name].to_numpy(dtype=bool)
                        coverage.append({'outer_fold':outer,'population':population,'role':r,'stratum':typ,'level':level,'gate':name,'queries':len(ix),'cases':pop.case_uid.nunique(),'groups':pop.association_group.nunique(),'supported':int(ok.sum()),'fallback':int((~ok).sum()),'window_coverage':float(ok.mean()),'group_case_window_equal_coverage':float(weights_query@ok)})
                    ep=rr.loc[ix,'ESS_only_primary'].to_numpy() & ~rr.loc[ix,'distance_only_Q99'].to_numpy()
                    coverage.append({'outer_fold':outer,'population':population,'role':r,'stratum':typ,'level':level,'gate':'HIGH_ESS_BUT_FAR_Q99','queries':len(ix),'cases':pop.case_uid.nunique(),'groups':pop.association_group.nunique(),'supported':int(ep.sum()),'fallback':int((~ep).sum()),'window_coverage':float(ep.mean()),'group_case_window_equal_coverage':float(weights_query@ep)})
            checks.append({'outer_fold':outer,'population':population,'queries':len(rr),'all_original_IDs':bool(np.array_equal(rr.original_cache_index,f.original_cache_index)),'fit_groups':len(gu),'fit_cases':len(cstarts),'fit_windows':len(ref)})
            print('DONE',outer,population,round(time.perf_counter()-started,1),flush=True)
    csv('condition_scalers.csv',pd.DataFrame(scales));csv('distance_thresholds_fit_normal_LOO.csv',pd.DataFrame(thresholds))
    csv('condition_support_distribution.csv',pd.DataFrame(all_summaries));csv('joint_gate_coverage.csv',pd.DataFrame(coverage))
    js('completion.json',{'status':'COMPLETE_V1_CSV_ONLY_C_KERNEL_SUPPORT','queries_per_fold_population':162069,'fold_population_jobs':checks,'seconds':time.perf_counter()-started,'no_models_latents_h_kernels_or_U_C_scores':True,'no_original_source_data_or_weights_read':True,'input_partition_verified':True,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__=='__main__':main()
