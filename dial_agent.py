"""Dial agent for the Iowa Gambling Task (PVL-Delta style).
Dials: reward sensitivity (gain weight, fixed scale 1), fear (lambda, loss weight relative to gain),
learning rate A, consistency c, outcome curvature alpha."""
import numpy as np
from scipy.optimize import minimize
GAIN=np.array([100,100,50,50.])
LOSS10={0:[150,200,250,300,350,0,0,0,0,0],1:[1250]+[0]*9,2:[50]*5+[0]*5,3:[250]+[0]*9}
def schedule(rng,n=100):
    s=np.zeros((4,n))
    for d in range(4):
        for k in range(0,n,10):
            s[d,k:k+10]=rng.permutation(LOSS10[d])
    return s
def simulate(p,rng,n=100):
    alpha,lam,A,c=p; th=3**c-1; E=np.zeros(4); ptr=np.zeros(4,int); sch=schedule(rng,n)
    ch=np.zeros(n,int); g=np.zeros(n); l=np.zeros(n)
    for t in range(n):
        z=th*E; pr=np.exp(z-z.max()); pr/=pr.sum(); d=rng.choice(4,p=pr)
        win=GAIN[d]/100; loss=sch[d,ptr[d]]/100; ptr[d]+=1
        u=win**alpha-lam*loss**alpha
        E[d]+=A*(u-E[d]); ch[t]=d; g[t]=win; l[t]=loss
    return ch,g,l
def nll(p,ch,g,l):
    alpha,lam,A,c=p; th=3**c-1; E=np.zeros(4); s=0
    for t in range(len(ch)):
        z=th*E; z-=z.max(); s+=z[ch[t]]-np.log(np.exp(z).sum())
        u=g[t]**alpha-lam*l[t]**alpha; E[ch[t]]+=A*(u-E[ch[t]])
    return -s
BOUNDS=[(0.05,1),(0.01,5),(0.01,1),(0.01,5)]
def fit(ch,g,l,rng,starts=6):
    best=None
    for _ in range(starts):
        x0=[rng.uniform(a,b) for a,b in BOUNDS]
        r=minimize(nll,x0,args=(ch,g,l),bounds=BOUNDS,method="L-BFGS-B")
        if best is None or r.fun<best.fun: best=r
    return best.x
def block_scores(ch):
    good=(ch>=2).astype(int); return np.array([2*good[i:i+20].sum()-20 for i in range(0,100,20)])
