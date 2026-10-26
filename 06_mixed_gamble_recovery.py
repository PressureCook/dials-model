"""Mixed gamble task: 50/50 win G or lose L, accept or reject. Gains and losses vary independently,
so reward and fear weights can be separated. 128 trials."""
import numpy as np
from scipy.optimize import minimize
G,L=np.meshgrid(np.arange(10,41,2)/10,np.arange(5,21,2)/10); G=G.ravel(); L=L.ravel()  # 16x8=128 offers
def p_acc(th,G,L):
    a,lam,c=th; v=.5*G**a-.5*lam*L**a; return 1/(1+np.exp(-c*v))
def fit(acc,rng):
    best=None
    for _ in range(5):
        x0=[rng.uniform(.3,1.2),rng.uniform(.5,4),rng.uniform(1,10)]
        f=lambda x: -np.sum(acc*np.log(p_acc(x,G,L)+1e-9)+(1-acc)*np.log(1-p_acc(x,G,L)+1e-9))
        r=minimize(f,x0,bounds=[(.2,1.5),(.1,8),(.1,30)],method="L-BFGS-B")
        if best is None or r.fun<best.fun: best=r
    return best.x
rng=np.random.default_rng(5); T=[];E=[]
for k in range(200):
    th=np.array([.8*np.exp(rng.normal(0,.2)),2*np.exp(rng.normal(0,.5)),5*np.exp(rng.normal(0,.4))])
    acc=(rng.random(len(G))<p_acc(th,G,L)).astype(float); T.append(th); E.append(fit(acc,rng))
T=np.array(T);E=np.array(E)
for i,n in enumerate(["outcome sensitivity","fear (loss weight)","consistency"]):
    print(n,"recovery r =",round(np.corrcoef(np.log(T[:,i]),np.log(E[:,i]))[0,1],2))
