"""Read completed results, profile them, and render auditable baseline figures."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SKILL=Path('C:/Users/23119/.agents/skills/scipilot-figure-skill/scripts')
sys.path.insert(0,str(SKILL))
from profile_data import profile_data,render_report
from setup_style import setup_style
from visual_qa import render_preview,audit_layout
from export_figure import export_figure

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research/2026-10-08/uav_baselines_v1'
NAMES={'pca':'PCA','gru':'GRU','persistence':'上一时刻预测'}
COLORS={'pca':'#0072B2','gru':'#D55E00','persistence':'#777777'}
MARKERS={'pca':'o','gru':'s','persistence':'^'}
STYLES={'pca':'-','gru':'--','persistence':':'}
ORDER=['pca','gru','persistence']


def main():
    args=argparse.ArgumentParser();args.add_argument('--export',action='store_true');opts=args.parse_args()
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    rows=json.loads((OUT/'case_metrics.json').read_text(encoding='utf-8'))
    meta=json.loads((OUT/'case_metadata.json').read_text(encoding='utf-8'))
    thresholds=json.loads((OUT/'thresholds.json').read_text(encoding='utf-8'))
    test={r['model']:r for r in summary if r['split']=='test'}
    (OUT/'result_data_profile.md').write_text(render_report(profile_data(str(OUT/'case_metrics.csv'),group_cols=['split','model'])),encoding='utf-8')
    style=setup_style(journal='general',lang='zh',use_sciplots=False)
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':9})
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    qa={};supplement=[]
    for model in ORDER:
        mr=[r for r in rows if r['split']=='test' and r['model']==model]
        delays=[d for r in mr for d in r['detected_delays_s']]
        normal=[r for r in mr if r['level3_label']=='normal']
        supplement.append(dict(model=model,events=481,detected_within_1s=sum(d<=1 for d in delays),
            detected_within_3s=sum(d<=3 for d in delays),detected_within_5s=sum(d<=5 for d in delays),
            normal_cases=len(normal),normal_cases_with_alarm=sum(r['false_alarm_events']>0 for r in normal)))
    pd.DataFrame(supplement).to_csv(OUT/'timely_detection.csv',index=False,encoding='utf-8-sig')

    def save(fig,name):
        fig.set_dpi(160)
        fig.get_layout_engine().set(w_pad=.1,h_pad=.1)
        render_preview(fig,str(figs/f'{name}_preview.png'),dpi=160)
        issues=audit_layout(fig);qa[name]=issues
        print(name,issues)
        if any(sev=='FAIL' for sev,msg in issues): raise RuntimeError('Figure layout failed')
        if opts.export:
            export_figure(fig,str(figs/name),formats=['svg','png'],size_inches=tuple(fig.get_size_inches()),dpi=300,tight=False,grayscale_preview=True)
        plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True)
    for i,model in enumerate(ORDER):
        y=2-i;r=test[model]
        for ax,key,ci,factor in [(axes[0],'event_recall','event_recall_95ci',100),(axes[1],'false_alarms_per_normal_hour','false_alarms_per_normal_hour_95ci',1)]:
            value=r[key]*factor;lo,hi=np.array(r[ci])*factor
            ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt=MARKERS[model],color=COLORS[model],capsize=3,markersize=6)
            ax.annotate(f'{value:.1f}',(value,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
    axes[0].set_yticks([2,1,0],[NAMES[m] for m in ORDER]);axes[0].set_xlim(0,100)
    axes[0].set_xlabel('标签起点后新报警比例 (%)');axes[0].set_title('(a) 严格事件检出率')
    axes[1].set_xlim(left=0);axes[1].set_xlabel('误报次数 / 正常小时');axes[1].set_title('(b) 正常时段误报')
    for ax in axes: ax.set_ylim(-.55,2.6);ax.grid(axis='x',alpha=.2)
    fig.suptitle('仿真测试：553 个案例，阈值固定于验证集',fontsize=11)
    save(fig,'test_comparison')

    # Illustrations selected from VALIDATION only; selection rules saved below.
    vr=[r for r in rows if r['split']=='val' and r['model']=='gru']
    false=max((r for r in vr if r['level3_label']=='normal'),key=lambda r:(r['false_alarm_events'],r['case_uid']))
    misses=sorted((r for r in vr if r['fault_events'] and not r['detected_events'] and r['level3_label']=='motor'),key=lambda r:r['case_uid'])
    miss=misses[0]
    hits=sorted((r for r in vr if r['detected_delays_s']),key=lambda r:(r['detected_delays_s'][0],r['case_uid']))
    hit=hits[len(hits)//2]
    chosen=[false,miss,hit]
    selected=[]
    fig,axes=plt.subplots(3,1,figsize=(7.2,6.0))
    panels=['正常案例中的误报','标签前已报警的电机案例','故障区间内产生报警的案例']
    for i,(ax,r) in enumerate(zip(axes,chosen)):
        metadata=next(m for m in meta if m['case_uid']==r['case_uid'])
        s=np.load(OUT/r['score_file'])
        for begin,end in metadata['intervals']: ax.axvspan(begin,end,color='#BBBBBB',alpha=.25)
        for model in ORDER:
            ax.plot(s['t'],np.maximum(s[model]/thresholds[model],1e-6),STYLES[model],color=COLORS[model],lw=1.,label=NAMES[model])
        ax.axhline(1,color='black',lw=.8,ls='-.')
        ax.set_yscale('log');ax.set_ylabel('分数 / 阈值（对数）')
        ax.set_title(f'({chr(97+i)}) {panels[i]} · {r["domain"]} / {r["flight_status"]}',loc='left')
        ax.set_xlabel('自记录开始的时间 (s)');ax.grid(alpha=.15)
        selected.append(dict(case_uid=r['case_uid'],panel=panels[i],selection=['maximum GRU false alarms among validation normal cases','first sorted validation motor miss','median validation detected-only GRU delay'][i],score_file=r['score_file']))
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,1.45),ncol=3,frameon=False,fontsize=8)
    save(fig,'validation_examples')
    (OUT/'figure_metadata.json').write_text(json.dumps(dict(style=style,qa=qa,examples=selected,
        comparison_caption='Dots: fixed-threshold test estimates. Error bars: 95% percentile CIs from 500 resamples of 277 SIL/HIL scenario groups; 553 cases, 481 fault intervals. No significance test. One GRU seed; intervals do not capture training variability.',
        examples_caption='Selected validation illustrations, not representative frequencies. Gray shading: processed fault-active intervals; horizontal line: threshold. Three consecutive decisions are required to alarm; a single crossing is not an alarm. Scores below 1e-6 of threshold are floored only for plotting. Linear-interpolated inputs: timing is exploratory.'),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
