"""Upbringing -> dials: synthetic people with links set from published effect sizes (ASSUMPTIONS).
True childhood factors: temperament (fearful, empathic, reward-eager), parental care, overprotection, adversity.
Adult dials (true) = temperament carry-over + parenting effects + everything else (genes, peers, chance).
Observed: retrospective self report (noisy), optional parent report of temperament, task-measured dials (noisy labels)."""
import numpy as np
from sklearn.linear_model import RidgeCV
TASK_ERR=np.array([0.21,0.50,0.75])     # guilt, fear, reward measurement error from the task integrator
def people(n,rng):
    # childhood factors: Tfear, Tguilt, Treward, care, overprotection, adversity
    C=np.eye(6); C[3,4]=C[4,3]=-0.3; C[3,5]=C[5,3]=-0.45; C[4,5]=C[5,4]=0.15; C[1,3]=C[3,1]=0.2; C[0,4]=C[4,0]=0.15
    F=rng.multivariate_normal(np.zeros(6),C,size=n); Tf,Tg,Tr,care,over,adv=F.T
    # adult dials (standardised): carry-over ~0.35, parenting effects ~0.1 to 0.2 (literature-sized)
    guilt = 0.35*Tg + 0.20*care - 0.10*adv
    fear  = 0.35*Tf + 0.10*over + 0.05*adv
    reward= 0.30*Tr + 0.05*adv - 0.05*care
    D=np.stack([guilt,fear,reward],1); D+= rng.normal(size=D.shape)*np.sqrt(1-D.var(0))   # rest of the variance
    # self report: 3 temperament items per dial (recall r~0.6 with truth), PBI care 12 + overprotection 13 items, ACE count
    def items(latent,k,load,noise_scale): return load*latent[:,None]+rng.normal(size=(n,k))*noise_scale
    recall=lambda x: 0.75*x+np.sqrt(1-0.75**2)*rng.normal(size=n)     # memory distortion of the whole factor
    temp=np.concatenate([items(recall(Tf),3,.7,.7),items(recall(Tg),3,.7,.7),items(recall(Tr),3,.7,.7)],1)
    pbi=np.concatenate([items(recall(care),12,.7,.7),items(recall(over),13,.6,.8)],1)
    ace=np.clip(np.round(np.exp(0.2+0.9*adv-0.4)*rng.gamma(2,.5,n)),0,10)[:,None]
    parent=np.concatenate([items(Tf,3,.7,.7),items(Tg,3,.7,.7),items(Tr,3,.7,.7)],1)   # parent's report, no recall distortion of the child
    measured=D+rng.normal(size=D.shape)*TASK_ERR
    return dict(true=D,measured=measured,self=np.concatenate([temp,pbi,ace],1),parent=parent,F=F)
if __name__=="__main__":
    rng=np.random.default_rng(1); test=people(20000,rng)
    # ceiling: perfect knowledge of the childhood factors
    from sklearn.linear_model import LinearRegression
    ceil=[np.corrcoef(LinearRegression().fit(test["F"],test["true"][:,i]).predict(test["F"]),test["true"][:,i])[0,1] for i in range(3)]
    print("ceiling (perfect childhood knowledge) r guilt/fear/reward:",np.round(ceil,2))
    print("ACE count distribution:",np.bincount(test["self"][:,-1].astype(int),minlength=11)[:8]/20000)
    for label,use_parent in [("self report only",False),("self + parent",True)]:
        Xte=np.concatenate([test["self"]]+([test["parent"]] if use_parent else []),1)
        for n in [50,100,200,400,800,1600,5000]:
            rs=[];rm=[]
            for rep in range(5):
                tr=people(n,np.random.default_rng(100+rep*7+n))
                Xtr=np.concatenate([tr["self"]]+([tr["parent"]] if use_parent else []),1)
                m=RidgeCV(alphas=np.logspace(-1,4,20)).fit(Xtr,tr["measured"])   # trained on task-measured dials
                p=m.predict(Xte)
                rs.append([np.corrcoef(p[:,i],test["true"][:,i])[0,1] for i in range(3)])
                rm.append([np.corrcoef(p[:,i],test["measured"][:,i])[0,1] for i in range(3)])
            print(f"{label:17s} n={n:5d}  r vs true dials {np.round(np.mean(rs,0),2)}  r vs task readings {np.round(np.mean(rm,0),2)}")
