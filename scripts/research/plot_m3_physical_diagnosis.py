"""Two descriptive M3 diagnostic figures from case statistics and native traces."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
sys.path.insert(0,'C:/Users/23119/.agents/skills/scipilot-figure-skill/scripts')
from visual_qa import audit_layout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    a=parser.parse_args();out=a.results/'figures'
    if out.exists():raise FileExistsError('Preserve existing figures')
    out.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    qa={}
    def export(fig,name):
        fig.canvas.draw();issues=audit_layout(fig)
        if any(s=='FAIL' for s,_ in issues):raise ValueError(issues)
        qa[name]=issues
        fig.savefig(out/(name+'.png'),dpi=300)
        ImageOps.grayscale(Image.open(out/(name+'.png'))).save(out/(name+'_grayscale.png'))
        fig.savefig(out/(name+'.svg'));plt.close(fig)
    nominal=pd.read_csv(a.results/'signed_case_AUC.csv');motion=pd.read_csv(a.results/'motion_conditioned_signed_AUC.csv')
    fig,axes=plt.subplots(1,3,figsize=(11,3.8));fig.subplots_adjust(left=.065,right=.99,bottom=.31,top=.82,wspace=.28)
    faults=['GPS','gyroscope','magnetometer'];labels=['GPS\nscenario','GPS\nlow-motion','Gyro\nscenario','Gyro\nlow-motion','Mag\nscenario','Mag\nlow-motion']
    for j,feature in enumerate(['O4','N3','N4']):
        ax=axes[j]
        for k,(split,domain,color,marker) in enumerate([('train','SIL','#0072B2','o'),('train','HIL','#56B4E9','s'),('val','SIL','#D55E00','^'),('val','HIL','#E69F00','D')]):
            vals=[]
            for fault in faults:
                n=nominal[(nominal.feature==feature)&(nominal.fault==fault)&(nominal.flight=='hover')&(nominal.split==split)&(nominal.domain==domain)]
                m=motion[(motion.feature==feature)&(motion.fault==fault)&(motion.flight=='hover')&(motion.observed_state=='low_motion_proxy')&(motion.split==split)&(motion.domain==domain)]
                vals.extend([n.signed_AUC.iloc[0],m.signed_AUC.iloc[0] if len(m) else np.nan])
            ax.scatter(np.arange(6)+(k-1.5)*.1,vals,color=color,marker=marker,s=25,label=f'{split.title()} {domain}')
        ax.axhline(.5,color='0.5',ls='--',lw=.8);ax.set_ylim(-.02,1.02);ax.set_xticks(range(6),labels,fontsize=7)
        ax.set_title(f'{chr(97+j)}  {feature}');ax.set_ylabel('Signed case-level AUC');ax.grid(axis='y',alpha=.2)
    handles,labelslegend=axes[0].get_legend_handles_labels();fig.legend(handles,labelslegend,ncol=4,loc='lower center',bbox_to_anchor=(.5,.09),frameon=False)
    fig.suptitle('Hover discrimination changes after matching observed low motion',y=.97)
    fig.text(.5,.02,'One median per case/state. No sign flips or confidence intervals; Val cells have only 3–5 fault cases and 6 normal cases.',ha='center',fontsize=7)
    export(fig,'hover_AUC_state_control')
    cases=['SIL__hover__gyroscope__TestCase_11_1003000010','SIL__hover__no_fault__TestCase_31_1019000030']
    fig,axes=plt.subplots(4,2,figsize=(10,8),sharex='col');fig.subplots_adjust(left=.10,right=.99,bottom=.12,top=.92,hspace=.25,wspace=.27)
    h=pd.read_csv(a.results/'hover_window_medians.csv')
    for col,key in enumerate(cases):
        f=pd.read_csv(a.results/'local_traces'/(key+'.csv'));t=f.crop_seconds.to_numpy();uid=f.case_uid.iloc[0]
        arrays=[f.vel_z.to_numpy(),np.linalg.norm(f[['gyro_x','gyro_y','gyro_z']],axis=1),f.N3.where(f.residual_valid).to_numpy(),f.N4.where(f.residual_valid).to_numpy()]
        titles=['Vertical velocity (m/s, NED)','Gyro norm (rad/s)','N3 signed (rad)','N4 signed (rad)']
        selected=h[h.case_uid.eq(uid)&h.comparison_group.eq('CN_new_FN' if col==0 else 'CN_new_normal_FP')]
        union=np.zeros(len(f),dtype=bool)
        for _,row in selected.iterrows():union[int(row.window_start_row):int(row.window_end_row)]=True
        for row,(values,label) in enumerate(zip(arrays,titles)):
            ax=axes[row,col];ax.plot(t,values,color='#0072B2' if col==0 else '#D55E00',lw=.7,ls='-' if col==0 else '--')
            for s,e in zip(np.flatnonzero(np.diff(np.r_[False,union].astype(int))==1),np.flatnonzero(np.diff(np.r_[union,False].astype(int))==-1)):
                ax.axvspan(t[s],t[e],color='0.85',alpha=.5,zorder=-2)
            if col==0:
                active=np.flatnonzero((f.reference_state==1)&f.label_valid)
                if len(active):ax.axvline(t[active[0]],color='0.3',ls=':',lw=.8)
            if row>=2:ax.set_yscale('symlog',linthresh=1e-4)
            ax.set_ylabel(label);ax.grid(alpha=.18)
        axes[0,col].set_title(('a  Gyroscope-fault case' if col==0 else 'b  Known-normal case')+'\n'+key.split('__')[-1],fontsize=9)
        axes[-1,col].set_xlabel('Original cropped grid time (s)')
    fig.text(.5,.055,'Grey: union of CN new-FN windows (left) / CN new-FP windows (right). Dotted: first valid fault label, not physical damage truth.',ha='center',fontsize=7)
    fig.text(.5,.025,'Unique source frames; no window repetition. N3/N4 use signed symlog axes; invalid residual positions are gaps, never filled.',ha='center',fontsize=7)
    export(fig,'native_hover_fault_and_normal')
    (out/'visual_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
