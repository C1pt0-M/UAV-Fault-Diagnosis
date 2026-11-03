"""Plot saved case-level candidate correlations and observed-state shifts."""
from pathlib import Path
import argparse,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from analyze_13_feature_screening import IDS


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--skill-scripts',type=Path,required=True);ap.add_argument('--export-svg',action='store_true');a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve figure outputs')
    sys.path.insert(0,str(a.skill_scripts));from visual_qa import audit_layout
    a.output.mkdir(parents=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    cases=pd.read_csv(a.results/'case_feature_medians.csv');all_groups=pd.read_csv(a.results/'feature_correlation_by_group.csv');qa=[]
    def finish(fig,name):
        issues=audit_layout(fig);qa.append({'figure':name,'issues':[list(x) for x in issues]})
        if any(severity=='FAIL' for severity,_ in issues):raise RuntimeError('Missing glyphs in '+name)
        fig.savefig(a.output/(name+'.png'),dpi=300)
        if a.export_svg:fig.savefig(a.output/(name+'.svg'))
        with Image.open(a.output/(name+'.png')) as image:image.convert('L').save(a.output/(name+'_grayscale.png'))
        plt.close(fig)
    # Primary matrix and exposed-Val confirmation; identical scale and IDs.
    fig,axes=plt.subplots(1,2,figsize=(12.,5.5),layout='constrained')
    for ax,scope,label in zip(axes,['train_case_medians','val_case_medians'],['a','b']):
        part=all_groups[(all_groups.scope==scope)&(all_groups.method=='spearman')];c=part.pivot(index='feature_a',columns='feature_b',values='correlation').loc[IDS,IDS].to_numpy()
        im=ax.imshow(c,vmin=-1,vmax=1,cmap='RdBu_r');n=(cases.split==('train' if scope.startswith('train') else 'val')).sum()
        ax.set_title(f'{label}  '+('Train' if scope.startswith('train') else 'Exposed Val')+f': {n} case medians',loc='left',fontweight='bold')
        ax.set_xticks(range(13),IDS,rotation=45,ha='right');ax.set_yticks(range(13),IDS)
        for i in range(13):
            for j in range(13):ax.text(j,i,f'{c[i,j]:.2f}',ha='center',va='center',fontsize=6.2,color='white' if abs(c[i,j])>.65 else 'black')
        ax.axhline(8.5,color='black',lw=.8);ax.axvline(8.5,color='black',lw=.8)
    fig.colorbar(im,ax=axes,shrink=.8,label='Spearman correlation (one median vector per case)')
    fig.suptitle('Candidate redundancy: signed full150 summaries; no independent-window significance',fontsize=10)
    finish(fig,'feature_correlation_heatmap')
    # Show every case contrast, not a mean bar or unearned confidence interval.
    contrasts=pd.read_csv(a.results/'normal_state_paired_shifts.csv');contrasts=contrasts[contrasts.paired & (contrasts.contrast=='low_motion_proxy minus climb_proxy')]
    fig,axes=plt.subplots(2,1,figsize=(10.,6.),sharex=True,sharey=True,layout='constrained')
    for ax,split,color in zip(axes,['train','val'],['#0072B2','#D55E00']):
        p=contrasts[contrasts.split==split]
        for i,name in enumerate(IDS):
            v=p[p.feature_id==name].sort_values('case_uid').paired_shift_in_train_normal_IQR.to_numpy();jitter=np.linspace(-.16,.16,len(v))
            ax.scatter(i+jitter,v,s=17,facecolors='none',edgecolors=color,linewidths=.8)
            ax.plot([i-.24,i+.24],[np.median(v)]*2,color='black',lw=1.3)
        ax.axhline(0,color='gray',lw=.7,linestyle='--');ax.set_ylabel('Shift / diagnostic Train-normal IQR')
        ax.set_title(f'{split}: each point is one paired normal case; n={p.case_uid.nunique()} (SIL/HIL pairs not independent)',loc='left',fontsize=9)
    axes[-1].set_xticks(range(13),IDS);fig.suptitle('Normal observed state: low motion minus climb (signal proxies, not directory phase)',fontsize=10)
    finish(fig,'normal_state_paired_shifts')
    (a.output/'figure_QA.json').write_text(json.dumps({'machine_audit':qa,'case_aggregation':'No window-level p-values, no CI, no sign flipping','primary_matrix_rows':28,'exposed_val_matrix_rows':18,'AI_visual_review':'Pending final-image inspection; original PNG previews reviewed before SVG export' if a.export_svg else 'PENDING_READ_TOOL','export_svg':bool(a.export_svg)},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(qa,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
