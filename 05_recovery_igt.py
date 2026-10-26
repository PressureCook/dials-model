import numpy as np, pandas as pd
from dial_agent import *
p0=np.load("p0.npy"); norm=pd.read_csv("igt_norms.csv"); sd=norm.ctrl_sd.values
def curve(p,N=1000,seed=99):
    rng=np.random.default_rng(seed); return np.array([block_scores(simulate(p,rng)[0]) for _ in range(N)])
B=curve(p0)
print("Alternative explanations for the same deficit:")
for name,i,f in [("learning rate x0.4",2,.4),("consistency x0.5",3,.5),("outcome sensitivity x0.5",0,.5)]:
    p=p0.copy(); p[i]*=f; print(name,((curve(p).mean(0)-B.mean(0))/sd).round(2))
# parameter recovery: 120 simulated people, dials drawn around the healthy setting
rng=np.random.default_rng(3); true=[];est=[]
for k in range(120):
    p=np.array([np.clip(p0[0]*np.exp(rng.normal(0,.3)),.05,1),p0[1]*np.exp(rng.normal(0,.5)),
                np.clip(p0[2]*np.exp(rng.normal(0,.5)),.01,1),p0[3]*np.exp(rng.normal(0,.3))])
    ch,g,l=simulate(p,rng); true.append(p); est.append(fit(ch,g,l,rng,starts=5))
true=np.array(true);est=np.array(est)
names=["outcome sensitivity","fear (loss weight)","learning rate","consistency"]
for i,n in enumerate(names):
    r=np.corrcoef(np.log(true[:,i]),np.log(est[:,i]))[0,1]; print(f"{n}: recovery r = {r:.2f}")
pd.DataFrame(np.hstack([true,est]),columns=[f"true_{n}" for n in names]+[f"est_{n}" for n in names]).to_csv("recovery.csv",index=False)
