import numpy as np, pandas as pd, json, torch
D="/root/autodl-tmp/causal_source_v1_20261031/unpacked/data/"
cols=json.load(open(D+"input_contract.json"))["columns"]
L=pd.read_csv(D+"ledgers/train_included_windows.csv")
x=torch.load(D+"cache/train.pt",map_location="cpu",weights_only=False)["x"]
mu=pd.DataFrame(x.mean(2).numpy(),columns=cols)
L["flight"]=L.case_uid.str.split("/").str[1]; L["t"]=L.window_end_row*.02
L["src"]=np.where(L.normal_origin=="normal_flight","indep",np.where(L.qualified_y==1,"fcase_fault","fcase_normal"))
L["pwm"]=mu[[f"actuator_out_{i}" for i in range(1,5)]].mean(1); L["vz"]=mu.vel_z; L["alt"]=mu.baro_alt_relative; L["vx"]=mu.vel_x; L["spd"]=np.hypot(mu.vel_x,mu.vel_y)
L["tb"]=pd.cut(L.t,[0,5,10,20,30,40,50,60,70,90,200])
for fl,vs in [("hover",["pwm","vz","alt"]),("acce",["spd","pwm"]),("velocity",["spd","pwm"]),("waypoint",["spd","pwm"])]:
    print(f"\n==== {fl}")
    print(L[L.flight==fl].pivot_table(index="tb",columns="src",values=vs,aggfunc="median",observed=True).round(2).to_string())
# hover: indep flights - per-flight last 5s vz and alt (landing?)
H=L[(L.flight=="hover")]
g=H.groupby("case_uid").agg(src=("src","first"),T=("t","max"))
print("\nhover flight length by src", g.groupby("src").T.describe().round(1).to_string())
last=H[H.t>H.groupby("case_uid").t.transform("max")-5]
print("hover last-5s median vz/alt/pwm by origin", last.assign(o=np.where(last.normal_origin=="normal_flight","indep","fcase")).groupby("o")[["vz","alt","pwm"]].median().round(2).to_string())
# early-window (t<10s) pwm per flight distributions, hover
e=H[H.t<10].groupby("case_uid").agg(o=("normal_origin",lambda s:"indep" if (s=="normal_flight").all() else "fcase"),pwm=("pwm","median"))
print("hover per-flight early pwm by origin", e.groupby("o").pwm.describe().round(1).to_string())
