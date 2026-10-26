"""Learn the SD3 measurement model from real data: loadings, uniquenesses, factor correlations, item thresholds."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
from scipy.stats import norm
import factor_analyzer.factor_analyzer as _f
_ca=_f.check_array
_f.check_array=lambda X,**kw: _ca(X,**{('ensure_all_finite' if k=='force_all_finite' else k):v for k,v in kw.items()})
from factor_analyzer import FactorAnalyzer
X=pd.read_csv("items_clean.csv"); items=list(X.columns)
fa=FactorAnalyzer(n_factors=3,rotation="oblimin",method="minres"); fa.fit(X)
L=fa.loadings_
# name factors by marker items
idx={ "coldness":int(np.argmax(np.abs(L[items.index("M5")]))), "status":int(np.argmax(np.abs(L[items.index("N1")]))),
      "fearlessness":int(np.argmax(np.abs(L[items.index("P7")]))) }
order=[idx["coldness"],idx["fearlessness"],idx["status"]]
assert len(set(order))==3, idx
L=L[:,order]; phi=fa.phi_[np.ix_(order,order)]
# sign: make marker loadings positive
for j,m in enumerate(["M5","P7","N1"]):
    if L[items.index(m),j]<0: L[:,j]*=-1; phi[j,:]*=-1; phi[:,j]*=-1
uniq=np.clip(1-np.sum((L@phi)*L,axis=1),0.05,1)
thr=[]
for c in items:
    cum=np.cumsum(X[c].value_counts(normalize=True).reindex([1,2,3,4,5],fill_value=0).values)[:-1]
    thr.append(norm.ppf(np.clip(cum,1e-4,1-1e-4)).tolist())
json.dump(dict(items=items,L=L.tolist(),phi=phi.tolist(),uniq=uniq.tolist(),thr=thr,
               mean=X.mean().tolist(),sd=X.std().tolist()),open("sd3_measurement.json","w"))
print("factor order: coldness, fearlessness, status"); print(np.round(phi,2))
print(pd.DataFrame(L,index=items,columns=["cold","fearless","status"]).round(2).T.to_string())
