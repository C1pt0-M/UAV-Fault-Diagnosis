"""Summarize completed v2 CSVs for the report; no model or new data input."""
import json
from pathlib import Path
import pandas as pd

OUT=Path(__file__).resolve().parent
REPORT=OUT.parents[1]/'docs/research/M4协议修订与无训练补充审查_v2_2026-11-05.md'


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows])


def main():
    s=pd.read_csv(OUT/'condition_support_distribution.csv');c=pd.read_csv(OUT/'joint_gate_coverage.csv')
    t=pd.read_csv(OUT/'distance_thresholds_fit_normal_LOO.csv')
    gate='supported_G15_C30_D10_Q99'
    outer=c[c.role.eq('outer_held_out')].copy()
    pooled=outer.groupby(['population','stratum','level','gate'],as_index=False)[['queries','supported','fallback']].sum()
    pooled['coverage']=pooled.supported/pooled.queries
    pooled.to_csv(OUT/'pooled_outer_support_coverage.csv',index=False,mode='x')
    parts=['### 5.1 ESS分布：外层查询（全部原ID）\n']
    rows=[]
    for pop in ['D1','D2','D3']:
        for fold in range(3):
            pp=s[s.population.eq(pop)&s.outer_fold.eq(fold)&s.role.eq('outer_held_out')&s.stratum.eq('all')].set_index('metric')
            ps=['p10','p25','p50','p75','p90']
            rows.append([pop,fold,' / '.join(f'{pp.loc["group_ESS",p]:.2f}' for p in ps),' / '.join(f'{pp.loc["case_ESS",p]:.2f}' for p in ps),f'{pp.loc["min_domain_case_ESS","minimum"]:.3f}'])
    parts.append(table(['人口','fold','Group ESS P10/P25/P50/P75/P90','Case ESS P10/P25/P50/P75/P90','最小域内Case ESS'],rows))
    parts.append('\nD1参考有677—679组，外层Group ESS中位647.51—650.54；D2参考88组，中位84.45—84.70；D3参考589—591组，中位560.03—561.90。**b_c=1、按维数平均距离的权重对多数查询很宽，并非只使用几个局部背景案例。**高ESS不自动证明精确工况匹配；它也提示条件化可能只作较弱重加权，是否有h判别增量尚未验证。本版不因此缩带宽寻找正面结果。极端查询ESS可降至约1，组数量大不保证局部权重分散。域质量、域内ESS和全部最小值/百分位保留在完整表。\n')
    parts.append('### 5.2 距离：正常LOO门槛和外层分布\n')
    rows=[]
    for pop in ['D1','D2','D3']:
        for fold in range(3):
            tt=t[t.population.eq(pop)&t.outer_fold.eq(fold)&t.distance.eq('nearest_group_distance')].set_index('percentile')
            ss=s[s.population.eq(pop)&s.outer_fold.eq(fold)&s.role.eq('outer_held_out')&s.stratum.eq('all')&s.metric.eq('nearest_group_distance')].iloc[0]
            rows.append([pop,fold,f'{tt.loc[95,"threshold"]:.4f}/{tt.loc[99,"threshold"]:.4f}',' / '.join(f'{ss[p]:.4f}' for p in ['p50','p90','p95','p99'])])
    parts.append(table(['人口','fold','inner-fit正常LOO Q95/Q99','outer组距离 P50/P90/P95/P99'],rows))
    parts.append('\n最近案例、最近5案例及正常/dev/outer各角色距离分布全部发布，不将代表点的均值误认为完整正常轨迹分布。各人口尺度不同，上表数值不能直接当跨人口物理距离排名。\n')
    parts.append('### 5.3 主门禁覆盖与回退\n')
    rows=[]
    for pop in ['D1','D2','D3']:
        for fold in range(3):
            cc=outer[outer.population.eq(pop)&outer.outer_fold.eq(fold)&outer.gate.eq(gate)]
            allrow=cc[cc.stratum.eq('all')].iloc[0]
            n=cc[cc.stratum.eq('label')&cc.level.eq('normal')].iloc[0]
            f=cc[cc.stratum.eq('label')&cc.level.eq('fault')].iloc[0]
            rows.append([pop,fold,f'{int(allrow.supported)}/{int(allrow.queries)} ({100*allrow.window_coverage:.3f}%)',int(allrow.fallback),f'{100*n.window_coverage:.3f}%',f'{100*f.window_coverage:.3f}%'])
    parts.append(table(['人口','fold','全部外层支持/分母','回退窗','正常覆盖','故障覆盖'],rows))
    parts.append('\n以下为每ID只进入一次outer查询后池化；不是把三折重复窗口相加。\n')
    rows=[]
    keys={}
    for pop in ['D1','D2','D3']:
        def get(typ,level,g=gate):return pooled[pooled.population.eq(pop)&pooled.stratum.eq(typ)&pooled.level.eq(level)&pooled.gate.eq(g)].iloc[0]
        ar=get('all','all');nr=get('label','normal');fr=get('label','fault');far=get('all','all','HIGH_ESS_BUT_FAR_Q99')
        rows.append([pop,f'{int(ar.supported)}/162069',f'{100*ar.coverage:.3f}%',f'{100*nr.coverage:.3f}%',f'{100*fr.coverage:.3f}%',f'{int(far.supported)} ({100*far.coverage:.3f}%)'])
        keys[pop]={'supported':int(ar.supported),'fallback':int(ar.fallback),'all_coverage':float(ar.coverage),'normal_coverage':float(nr.coverage),'fault_coverage':float(fr.coverage),'high_ESS_but_far':int(far.supported)}
    parts.append(table(['人口','支持窗','全部覆盖','正常覆盖','故障覆盖','ESS通过但距离不通过'],rows))
    parts.append('\nD2覆盖较高不证明它的正常参考更真实或诊断更好；人口尺度、正常范围和组代表几何都会影响门禁。本版未据这些outer数字更换主人口或门槛。\n')
    parts.append('### 5.4 D1主提案的状态与域覆盖\n')
    rows=[]
    for typ in ['domain','state','fault_directory']:
        for _,r in pooled[pooled.population.eq('D1')&pooled.stratum.eq(typ)&pooled.gate.eq(gate)].iterrows():
            rows.append([typ,r.level,f'{int(r.supported)}/{int(r.queries)}',f'{100*r.coverage:.3f}%'])
    parts.append(table(['分层','类别','支持窗/全部该层窗','覆盖'],rows))
    parts.append('\n故障目录行包含该案例未激活正常部分，**不是纯故障支持率**；严格故障×域分母另见[交叉覆盖](../../artifacts/M4_protocol_review_v2_20261105/heldout_fault_domain_coverage.csv)。正常来源另见[来源覆盖](../../artifacts/M4_protocol_review_v2_20261105/heldout_normal_origin_coverage.csv)。下降1344窗全部回退，未删分母；每域/状态/故障及来源的其余人口数据在完整覆盖表。\n')
    parts.append('### 5.5 预登记门槛敏感性（不选最佳）\n')
    rows=[]
    for pop in ['D1','D2','D3']:
        for _,r in pooled[pooled.population.eq(pop)&pooled.stratum.eq('all')&pooled.gate.str.startswith('supported_')].iterrows():
            rows.append([pop,r.gate.replace('supported_',''),f'{100*r.coverage:.3f}%',int(r.fallback)])
    parts.append(table(['人口','ESS/距离组合','外层覆盖','回退窗'],rows))
    block='\n\n'.join(parts)+'\n'
    with (OUT/'support_report_tables.md').open('x',encoding='utf-8') as stream:stream.write(block)
    with (OUT/'support_key_findings.json').open('x',encoding='utf-8') as stream:json.dump(keys,stream,ensure_ascii=False,indent=2)
    text=REPORT.read_text(encoding='utf-8');marker='<!-- V2_NUMERIC_SUPPORT_TABLES -->'
    assert text.count(marker)==1
    REPORT.write_text(text.replace(marker,block),encoding='utf-8')
    print(json.dumps(keys,ensure_ascii=False))


if __name__=='__main__':main()
