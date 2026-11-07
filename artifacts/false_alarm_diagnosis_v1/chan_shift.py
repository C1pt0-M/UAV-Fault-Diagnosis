import numpy as np, pandas as pd, json, torch
D="/root/autodl-tmp/causal_source_v1_20261031/unpacked/data/"
R="/root/autodl-tmp/UAV-Fault-Diagnosis/artifacts/t_selection_mstca_v1/cv/"
cols=json.load(open(D+"input_contract.json"))["columns"]
L=pd.read_csv(D+"ledgers/train_included_windows.csv")
d=torch.load(D+"cache/train.pt",map_location="cpu",weights_only=False)
assert np.array_equal(d["original_cache_index"].numpy(),L.original_cache_index.to_numpy())
x=d["x"]; mu=x.mean(2).numpy(); sd=x.std(2).numpy()
F=pd.DataFrame(np.c_[mu,sd],columns=[c+":m" for c in cols]+[c+":s" for c in cols])
L["flight"]=L.case_uid.str.split("/").str[1]; L["t"]=L.window_end_row*.02
L["kind"]=np.where(L.qualified_y==1,"fault",np.where(L.normal_origin=="normal_flight","indep","prefault"))
# mean T0 OOF score over seeds
P=np.zeros(len(L))
for seed in [42,43,44,45,46]:
    s=[]
    for f in range(3):
        z=np.load(R+f"seed{seed}_fold{f}/heldout_predictions.npz"); e=json.load(open(R+f"seed{seed}_fold{f}/selections.json"))["T0"]["epoch"]
        s.append(pd.Series(z["probabilities"][e-1].astype(float),index=z["original_cache_index"]))
    P+=pd.concat(s).reindex(L.original_cache_index).to_numpy()/5
L["p"]=P
def auc(a,b):
    v=np.r_[a,b]; r=pd.Series(v).rank().to_numpy(); n1=len(a); n2=len(b)
    return (r[:n1].sum()-n1*(n1+1)/2)/(n1*n2)
on=L[L.fault_onset_row.notna()].groupby("case_uid").fault_onset_row.first()*.02
on=on.groupby(L.groupby("case_uid").flight.first()).median()
res={}
for fl in sorted(L.flight.unique()):
    T=on[fl]; m=L.flight==fl
    EN=m&(L.kind=="indep")&(L.t<T); LN=m&(L.kind=="indep")&(L.t>T+5)
    PF=m&(L.kind=="prefault"); LF=m&(L.kind=="fault")&(L.t>T+5)
    out={}
    for c in F.columns:
        f=F[c].to_numpy()
        out[c]=dict(EN_LN=auc(f[LN],f[EN]), PF_LN=auc(f[LN],f[PF]), PF_EN=auc(f[EN],f[PF]), LF_LN=auc(f[LF],f[LN]),
                    rho_p_LN=pd.Series(f[LN]).corr(pd.Series(L.p[LN].to_numpy()),method="spearman"))
    o=pd.DataFrame(out).T
    for k in ["EN_LN","PF_LN","PF_EN","LF_LN"]: o[k+"_d"]=(o[k]-.5).abs()
    print(f"\n===== {fl}  onset median {T:.1f}s  n EN {EN.sum()} LN {LN.sum()} PF {PF.sum()} LF {LF.sum()}; mean p EN {L.p[EN].mean():.3f} LN {L.p[LN].mean():.3f} PF {L.p[PF].mean():.3f}")
    print("-- top: late indep vs early indep (time drift in normal flight) --")
    print(o.sort_values("EN_LN_d",ascending=False).head(8)[["EN_LN","PF_LN","PF_EN","LF_LN","rho_p_LN"]].round(3).to_string())
    print("-- top: late indep vs prefault normal --")
    print(o.sort_values("PF_LN_d",ascending=False).head(8)[["EN_LN","PF_LN","PF_EN","LF_LN","rho_p_LN"]].round(3).to_string())
    print("-- top: early indep vs prefault (recording shift at matched time) --")
    print(o.sort_values("PF_EN_d",ascending=False).head(5)[["EN_LN","PF_LN","PF_EN","LF_LN","rho_p_LN"]].round(3).to_string())
    print("-- top |rho| score vs feature within late indep --")
    print(o.reindex(o.rho_p_LN.abs().sort_values(ascending=False).index).head(6)[["EN_LN","PF_LN","LF_LN","rho_p_LN"]].round(3).to_string())
    res[fl]=o.round(4).to_dict()
json.dump(res,open("/tmp/chan_shift.json","w"))
