"""Provisional upbringing predictor for the app: linear model on 6 scale scores, fitted to 60k synthetic people.
App features (z-scored): temperament fear, temperament guilt, temperament reward (means of 3 items, 1-5),
PBI care (0-36), PBI overprotection (0-39), ACE count (0-10)."""
import numpy as np, json
from sklearn.linear_model import Ridge
from upbringing_sim import people
d=people(60000,np.random.default_rng(9)); s=d["self"]
X=np.stack([s[:,0:3].mean(1),s[:,3:6].mean(1),s[:,6:9].mean(1),s[:,9:21].mean(1),s[:,21:34].mean(1),s[:,34]],1)
mu,sd=X.mean(0),X.std(0); Z=(X-mu)/sd
m=Ridge(1.0).fit(Z,d["true"]); pred=m.predict(Z); resid=(d["true"]-pred).std(0)
r=[np.corrcoef(pred[:,i],d["true"][:,i])[0,1] for i in range(3)]
names=["temp_fear","temp_guilt","temp_reward","care","overprotection","ace"]
print("coef (rows guilt/fear/reward):"); print(np.round(m.coef_,3)); print("r",np.round(r,2),"resid sd",np.round(resid,2))
# app-side normalisation: real-scale norms (assumed) for the questionnaire scores
norms={"temp_fear":[3.0,0.9],"temp_guilt":[3.6,0.8],"temp_reward":[3.0,0.9],"care":[27.0,7.0],"overprotection":[13.0,7.5],"ace":[1.2,1.6]}
json.dump(dict(features=names,coef=m.coef_.round(4).tolist(),intercept=m.intercept_.round(4).tolist(),resid_sd=resid.round(3).tolist(),
               norms=norms,r_true=[round(float(v),2) for v in r]),open("upbringing_model.json","w"),indent=1)
