"""Create traceable13-dimensional evidence table and provisional judgments.

Classifications are explicit research priorities, not automated deletion rules.
Every numerical entry comes from the frozen bounded diagnostics.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json
import numpy as np
import pandas as pd
from analyze_13_feature_screening import IDS,sha,read

JUDGMENTS={
'O1':('B','优先作输出离散代表，和O2替代对照','motor固定案例排序强；与O2/O5高度相关且正常爬升→低运动漂移，尚无MSTCA单维增量。'),
'O2':('B','作为O1的同维数替代候选，不默认二者都加','全局与O1高度相关，但部分故障内相关下降；高相关不足以证明可替代，不能仅据此删除。'),
'O3':('B','旧特征核心组候选，配O3/O4同容量比较','统计定义可复现，motor有条件排序；控制槽不等同电机目标，正常状态依赖，不能沿用跟踪残差物理解释。'),
'O4':('C','首轮核心不纳入，保留全9和替换O3的探针对照','Train与O3较相关但Val/故障内不稳定；正常漂移及motor响应方向改变；C是预算优先级，不是信息无用或最终删除。'),
'O5':('C','首轮核心不纳入，以完整旧9组保留反证机会','77.6%所选帧贡献触发z_acc截断，重叠输出离散、状态漂移和正常尾值；motor仍有排序信息，不能按物理命名差单独否定。'),
'O6':('B','测试加速度非线性代理的条件增量','Train所选加速度故障响应强但Val弱；正常机动也可大，基础26已有acc_norm，不能当新增传感器。'),
'O7':('B','保留活动量候选，须看正常机动与故障模式','所选val motor强排序，陀螺仪fault方向和train模式不同；正常旋转本身明显上升，非统一异常检测量。'),
'O8':('B','作为有barometer排序的统计代理比较，不称物理残差','物理量纲/差分缺dt问题保留；所选barometer有条件排序，正常状态漂移大，与N1不等价，禁止原位修公式。'),
'O9':('B','姿态统计候选，需未来同容量增量和姿态符号检查','所选val motor排序强、爬升→低运动中位漂移小，但机动/域/姿态参考依赖大；不当物理能量或几何不变量。'),
'N1':('B','垂向残差独立块，和姿态三轴分开测试','物理单位/因果定义及较高独立覆盖；同状态故障响应弱且模式可变，不能把自洽当健康；数据源不同于N2-4。'),
'N2':('B','姿态块保留x轴，若块有益再逐轴移除','有条件有方向信息、不同目录/域不一；与另两轴无普遍高相关，不能以范数替代或证明其独立增量。'),
'N3':('B','姿态块保留y轴，保留正常大峰值','部分正常case/frame有大修正峰；真实状态依赖和共享估计影响未分开，不能按物理定义直接A类。'),
'N4':('B','姿态块保留z轴并预设方向/域敏感核验','磁力计所选Val存在稳定反向排序，同时其他故障域方向可变；绝不事后翻符号；三轴不宜只取模长。')}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['results','summaries','historical','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve evidence output')
    a.output.mkdir(parents=True)
    cards=read(a.historical/'results/feature_cards.json');cards={x['id']:x for x in cards};valid=pd.read_csv(a.results/'feature_validity.csv').set_index('feature_id')
    redundancy=pd.read_csv(a.results/'redundancy_pairs.csv');fault=pd.read_csv(a.summaries/'fault_response_case_aggregates.csv');stability=pd.read_csv(a.results/'normal_state_paired_shifts.csv')
    reference=read(a.results/'diagnostic_reference.json');rows=[]
    for name in IDS:
        card=cards[name];v=valid.loc[name];p=redundancy[(redundancy.feature_a==name)|(redundancy.feature_b==name)];largest=p.iloc[p.train_case_spearman.abs().argmax()];other=largest.feature_b if largest.feature_a==name else largest.feature_a
        paired=stability[(stability.feature_id==name)&(stability.paired)&(stability.contrast=='low_motion_proxy minus climb_proxy')]
        normal_shifts={split:float(g.paired_shift_in_train_normal_IQR.median()) for split,g in paired.groupby('split')}
        response=fault[(fault.feature_id==name)&(fault.split=='val')];category,priority,reason=JUDGMENTS[name]
        row={'feature_id':name,'historical_key':card['feature_key'],'name':card['name'],'formula':card['formula'],
             'implementation':'packages/uav_fault/uav_fault/models/cnn/lstm.py::OnlineKGMSTCA._relation_features' if name.startswith('O') else 'scripts/research/validate_kinematic_residuals.py::compute_residuals; continuous_kinematic_residuals.py::compute_continuous/ContinuousResidualStream',
             'input_sources':'Frozen base26 source-train z-score channels; deterministic transform, no extra sensor information' if name.startswith('O') else 'native available baro+velocity' if name=='N1' else 'native available attitude quaternion+gyro; same200ms ordered rotation integration',
             'formula_evaluation':card['formula_scope'],'physical_unit':card['unit'],
             'normalization':'Base26 frozen train z-score first; old9 no separate scaling/clipping except original O5 clamp' if name.startswith('O') else 'Physical raw residual saved; M1 separate74700 source-train known-normal windows mean/std, never Val fit',
             'causal_availability':'At window end under base mask; O8 first time diff=0 is original formula; not native synchronization truth' if name.startswith('O') else '200ms past delivered-stream support, observation/available/age/reset/device and full150 masks; no future or per-window history restart',
             'validity_scope':'Only fixed46 source cases; not all3135 cases for old9',
             'base_admitted_windows':int(v.base_admitted_windows),'independent_valid_windows':int(v.independent_valid_windows),'own_full150_valid_fraction':float(v.own_full150_valid_fraction),
             'raw_frame_contribution_min':float(v.raw_min),'raw_frame_contribution_q95':float(v.raw_q95),'raw_frame_contribution_max':float(v.raw_max),
             'normal_raw_frame_contribution_min':float(v.normal_raw_min),'normal_raw_frame_contribution_max':float(v.normal_raw_max),
             'all_admitted_values_finite':True,'normal_distribution_file':'results/feature_distributions.csv and feature_normal_stability.csv; frame values and window-median distributions differ',
             'fault_distribution_file':'results/feature_fault_response.csv:312 per-case/feature rows, fixed signed AUC, pre-active effect, state controls with missing coverage',
             'val_state_matched_auc_by_fault':json.dumps({x.subtype:float(x.state_matched_auc_case_mean) for _,x in response.iterrows()},ensure_ascii=False),
             'normal_low_motion_minus_climb_train_IQR':normal_shifts.get('train'),'normal_low_motion_minus_climb_val_IQR':normal_shifts.get('val'),
             'normal_stability_limit':'Signal-derived proxies; Train16normal cases/Val6hover only; no descent eligible, little maneuver evidence, faults can change state',
             'largest_abs_train_case_correlation_partner':other,'largest_abs_train_case_spearman':float(largest.train_case_spearman),
             'domain_consistency':'Per-case/SIL-HIL signed responses and normal matched shifts retained; no universal consistency certified',
             'redundancy_scope':'One train-case median vector, n28; supplementary pair-collapse/within-case/state/class CSVs. Not input information conditional on M0.',
             'incremental_value_vs_M0':'UNPROVEN_FOR_THIS_DIMENSION; M0/M1/M2 group architecture/population/pretraining differ',
             'evidence_strength':'Finite calculability and descriptive responses; low evidence for feature-level model increment or fault-intensity robustness',
             'provisional_class':category,'next_experiment_priority':priority,'decision_reason':reason,'final_keep_or_delete':'NOT_DECIDED'}
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.output/'feature_evidence.csv',index=False)
    result={'status':'COMPLETE_PROVISIONAL13_FEATURE_SCREENING_NOT_FINAL_SELECTION','created_at':datetime.now(timezone.utc).isoformat(),
            'candidates':rows,'class_counts':{str(k):int(v) for k,v in pd.Series([r['provisional_class'] for r in rows]).value_counts().items()},
            'A_reason_empty':'No individual dimension has sufficiently controlled recall-preserving incremental evidence; physics and AUC do not substitute.',
            'D_reason_empty':'Each dimension has formula, finite validity and bounded per-case observations, enough for provisional B/C research priority; intensity and cross-condition incremental conclusions remain insufficient for all13.',
            'source':'Fixed46 case full150 new diagnostics plus recovered historical140 summaries; never concatenate the two protocols in a single statistic.',
            'provisional_old_core':['O1','O3','O6','O7','O8','O9'],'alternative_output_representative':'O2 swaps O1; not automatically additive',
            'reserved_old_features':['O4','O5'],'new_blocks':[['N1'],['N2','N3','N4']],
            'limitations':['No individual M0-increment proof.','Val repeatedly exposed, candidate proposal exploratory.','Only24 hover fault cases,2per split/type with SIL/HIL pairing; no independent intensity replication.','Cases/time windows not independent; no p-values.','Normal16Train+6Val cases with small observed-state groups; no eligible descent.'],
            'M3_executed':False,'server_GPU_test_Real_access':False,'Git_commit_or_push':False}
    (a.output/'feature_screening_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('FEATURE_EVIDENCE_ROWS',len(rows),'CLASSES',result['class_counts'])


if __name__=='__main__':main()
