"""Guilt dial: harm aversion task (after Crockett et al., 2014).
Each trial: choose between a 'more money, more shocks' option and a 'less money, fewer shocks' option.
Shocks go either to the participant (self) or to an anonymous other person.
Value difference = (1 - kappa) * money_diff - kappa * shock_diff. kappa_other is the guilt dial."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
def design(rng,n):
    dm=rng.integers(1,20,n)/2.0   # extra money, 0.5 to 9.5
    ds=rng.integers(1,20,n)       # extra shocks, 1 to 19
    return dm/10, ds/10           # rescale to similar ranges
def p_harm(k,g,dm,ds): return expit(g*((1-k)*dm-k*ds))
def simulate(ks,ko,g,rng,n=80):
    out=[]
    for k,who in [(ks,0),(ko,1)]:
        dm,ds=design(rng,n); y=(rng.random(n)<p_harm(k,g,dm,ds)).astype(float); out.append((dm,ds,y,np.full(n,who)))
    return [np.concatenate(z) for z in zip(*out)]
def fit(dm,ds,y,who,rng):
    def nll(x):
        ks,ko,lg=expit(x[0]),expit(x[1]),np.exp(x[2]); k=np.where(who==1,ko,ks)
        p=np.clip(p_harm(k,lg,dm,ds),1e-9,1-1e-9); return -np.sum(y*np.log(p)+(1-y)*np.log(1-p))
    best=min((minimize(nll,[rng.normal(),rng.normal(),rng.normal(2,.5)],method="L-BFGS-B") for _ in range(4)),key=lambda r:r.fun)
    return expit(best.x[0]),expit(best.x[1]),np.exp(best.x[2])
if __name__=="__main__":
    rng=np.random.default_rng(11); T=[];E=[]
    for n in [40,80]:
        T=[];E=[]
        for i in range(200):
            ks=expit(rng.normal(-0.4,0.8)); ko=expit(rng.normal(0,0.8)); g=np.exp(rng.normal(2.3,.4))
            dm,ds,y,who=simulate(ks,ko,g,rng,n); T.append((ks,ko,g)); E.append(fit(dm,ds,y,who,rng))
        T=np.array(T);E=np.array(E)
        r=[np.corrcoef(T[:,i],E[:,i])[0,1] for i in range(3)]
        hyper=np.corrcoef(T[:,1]-T[:,0],E[:,1]-E[:,0])[0,1]
        print(f"{n} trials per recipient ({2*n} total): self harm aversion r={r[0]:.2f}, guilt (other) r={r[1]:.2f}, consistency r={r[2]:.2f}, other minus self r={hyper:.2f}")
    # what a low guilt dial looks like behaviourally
    for ko in [.6,.4,.2,.05]:
        dm,ds=design(np.random.default_rng(0),5000); print(f"kappa_other {ko}: chooses to shock the other for money on {p_harm(ko,10,dm,ds).mean():.0%} of trials")
