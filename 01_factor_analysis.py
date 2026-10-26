import sys, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
import factor_analyzer.factor_analyzer as _f
_ca=_f.check_array
_f.check_array=lambda X,**kw: _ca(X,**{('ensure_all_finite' if k=='force_all_finite' else k):v for k,v in kw.items()})
df = pd.read_csv(sys.argv[1])
items=[f"{t}{i}" for t in "MNP" for i in range(1,10)]
REV=["N2","N6","N8","P2","P7"]
df=df[(df.flag_straightline==0)&(df.flag_lowvar==0)].copy()
X=df[items].astype(float).copy()
for c in REV: X[c]=6-X[c]
print("n",len(X)); print("KMO", round(calculate_kmo(X)[1],3))
def alpha(d):
    k=d.shape[1]; return k/(k-1)*(1-d.var().sum()/d.sum(axis=1).var())
for t in "MNP": print(t,"alpha",round(alpha(X[[c for c in items if c[0]==t]]),3))
ev=np.linalg.eigvalsh(X.corr().values)[::-1]; print("eigen",ev[:8].round(2))
# parallel analysis
rng=np.random.default_rng(0); sims=[]
for _ in range(30):
    R=rng.normal(size=X.shape); sims.append(np.linalg.eigvalsh(np.corrcoef(R.T))[::-1])
print("parallel 95pct", np.percentile(sims,95,axis=0)[:8].round(2))
for k in [3,4,5,6]:
    fa=FactorAnalyzer(n_factors=k, rotation="oblimin", method="minres"); fa.fit(X)
    L=pd.DataFrame(fa.loadings_,index=items).round(2)
    print(f"\n=== {k} factors; var explained {fa.get_factor_variance()[2][-1]:.2f}")
    L["top"]=L.abs().idxmax(axis=1)
    print(L.to_string())
    if hasattr(fa,"phi_") and fa.phi_ is not None: print("factor corr\n",np.round(fa.phi_,2))
X.to_csv("items_clean.csv",index=False)
