import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.mixture import GaussianMixture
X=pd.read_csv("items_clean.csv")
def alpha(d):
    k=d.shape[1]; return k/(k-1)*(1-d.var().sum()/d.sum(axis=1).var())
DIALS={"coldness":["M2","M5","M6","M8","M9","P1","P3","P5","P6","P9","M1","M7"],
       "status_drive":["N1","N2","N3","N4","N5","N6","N7","N8"],
       "fearlessness":["P2","P4","P7"]}
D=pd.DataFrame({k:X[v].mean(axis=1) for k,v in DIALS.items()})
for k,v in DIALS.items(): print(k,"alpha",round(alpha(X[v]),2))
Z=(D-D.mean())/D.std()
print(Z.corr().round(2))
psych=X[[f"P{i}" for i in range(1,10)]].mean(axis=1)
# how much of SD3 psychopathy does each dial explain
from sklearn.linear_model import LinearRegression
for cols in [["coldness"],["fearlessness"],["coldness","fearlessness"],["coldness","fearlessness","status_drive"]]:
    r2=LinearRegression().fit(Z[cols],psych).score(Z[cols],psych); print(cols,"R2",round(r2,3))
# top 5% "high psychopathy" group: where do they sit?
hi=psych>=psych.quantile(.95)
print("hi n",hi.sum(), "cut",round(psych.quantile(.95),2))
print(Z[hi].mean().round(2))
both=(Z.coldness>1)&(Z.fearlessness>1)
print("P(hi | cold>1sd & fearless>1sd)",round(hi[both].mean(),3),"n",both.sum())
print("P(hi | cold>1sd only)",round(hi[(Z.coldness>1)&(Z.fearlessness<=0)].mean(),3))
print("P(hi | fearless>1sd only)",round(hi[(Z.fearlessness>1)&(Z.coldness<=0)].mean(),3))
print("share of hi group with both >0.5sd", round(((Z.coldness>.5)&(Z.fearlessness>.5))[hi].mean(),3))
# dimensional vs categorical: GMM BIC on dial scores (jitter-free)
for k in range(1,6):
    g=GaussianMixture(k,covariance_type="full",random_state=0,n_init=3).fit(Z)
    print("GMM",k,"BIC",round(g.bic(Z)))
# skew of distributions
print(Z.skew().round(2))
# grid for heatmap: P(hi) by cold x fearless deciles
cb=pd.qcut(Z.coldness.rank(method="first"),10,labels=False); fb=pd.qcut(Z.fearlessness.rank(method="first"),10,labels=False)
grid=pd.crosstab(cb,fb,values=hi.astype(int),aggfunc="mean").round(3)
print(grid)
grid.to_csv("grid.csv")
out=Z.copy(); out["sd3_psych"]=psych; out["hi"]=hi.astype(int); out.to_csv("dial_scores.csv",index=False)
