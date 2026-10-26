import numpy as np, pandas as pd
from scipy.optimize import minimize
from dial_agent import *
norm=pd.read_csv("igt_norms.csv"); target=norm.ctrl_mean.values; sd=norm.ctrl_sd.values
def curve(p,N=300,seed=7):
    rng=np.random.default_rng(seed)
    return np.array([block_scores(simulate(p,rng)[0]) for _ in range(N)])
def loss(x):
    p=[np.clip(x[0],.05,1),np.clip(x[1],.01,5),np.clip(x[2],.01,1),np.clip(x[3],.01,5)]
    B=curve(p,150); return ((B.mean(0)-target)**2).sum()+((B.std(0)-sd)**2).sum()
best=None
for x0 in [[.3,1.5,.15,1],[.5,2.5,.05,.7],[.8,1,.05,.5],[.5,1.5,.3,.4]]:
    r=minimize(loss,x0,method="Nelder-Mead",options=dict(maxiter=120,xatol=1e-2,fatol=.5))
    print(np.round(r.x,3),round(r.fun,1))
    if best is None or r.fun<best.fun: best=r
p0=best.x; B=curve(p0,1000,seed=99)
print("healthy setting alpha,lambda,A,c =",p0.round(3))
print("target",target,"\nmodel ",B.mean(0).round(2),"\nmodel sd",B.std(0).round(2),"real sd",sd)
rows=[]
for f in [1,.75,.5,.3]:
    Bl=curve((p0[0],p0[1]*f,p0[2],p0[3]),1000,seed=99)  # common random numbers
    rows.append([f,round(p0[1]*f,2)]+list(((Bl.mean(0)-B.mean(0))/sd).round(2)))
print(pd.DataFrame(rows,columns=["fear_mult","lambda","g_b1","g_b2","g_b3","g_b4","g_b5"]).to_string(index=False))
rows=[]
for f in [1.5,2]:
    Bl=curve((p0[0],p0[1],p0[2]*1,p0[3]),1000,seed=99)
np.save("p0.npy",p0)
