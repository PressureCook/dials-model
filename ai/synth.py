"""Synthetic people for training the dials integrator.
Each person has three latent dials (standard units, correlated as in the real SD3 data):
  guilt  = minus coldness, fear = minus fearlessness, reward = status drive.
They then (a) answer the 27 SD3 items, (b) do 128 mixed gambles, (c) do 160 harm aversion choices.
Task links (ASSUMPTIONS, to be replaced by pilot data):
  fear   -> loss weight lambda = 2 * exp(0.5 * fear)
  reward -> gain weight w      = exp(0.3 * reward)
  guilt  -> harm aversion for others kappa_other = expit(0.8 * guilt)"""
import numpy as np, json
from scipy.special import expit
M=json.load(open("sd3_measurement.json"))
Lm=np.array(M["L"]); PHI=np.array(M["phi"]); UNIQ=np.array(M["uniq"]); THR=[np.array(t) for t in M["thr"]]
_G,_L=np.meshgrid(np.arange(10,41,2)/10,np.geomspace(0.5,6,8)); GG=_G.ravel(); GL=_L.ravel()   # 128 gambles, losses up to 6 (v2: harsher offers)
_r=np.random.default_rng(2026); HDM=_r.integers(1,20,80)/20; HDS=_r.integers(1,20,80)/10    # 80 harm offers, used for self and other
N_ITEMS,N_GAMB,N_HARM=27,128,160
def people(n,rng,rho=0.3,spread=1.0):
    # rho = how strongly self report tracks the true (behavioural) dial
    f=spread*rng.multivariate_normal(np.zeros(3),PHI,size=n)   # true dials in cold, fearless, status orientation
    cold,fearless,status=f.T
    dials=np.stack([-cold,-fearless,status],1)                  # guilt, fear, reward
    # questionnaire answers come from a self image that only partly matches the true dials
    f_self=rho*f+np.sqrt(1-rho**2)*rng.multivariate_normal(np.zeros(3),PHI,size=n)
    z=f_self@Lm.T+rng.normal(size=(n,27))*np.sqrt(UNIQ)
    items=np.stack([1+np.searchsorted(THR[j],z[:,j]) for j in range(27)],1).astype(float)
    # mixed gambles
    guilt,fear,reward=dials.T
    lam=2*np.exp(0.5*fear); w=np.exp(0.3*reward)
    alpha=0.8*np.exp(rng.normal(0,.2,n)); c=5*np.exp(rng.normal(0,.4,n))
    v=0.5*w[:,None]*GG[None]**alpha[:,None]-0.5*lam[:,None]*GL[None]**alpha[:,None]
    gamb=(rng.random((n,N_GAMB))<expit(c[:,None]*v)).astype(float)
    # harm aversion: self block then other block
    ko=expit(0.8*guilt); ks=expit(rng.normal(-0.4,0.8,n)); g=np.exp(rng.normal(2.3,.4,n))
    def block(k): return (rng.random((n,80))<expit(g[:,None]*((1-k[:,None])*HDM[None]-k[:,None]*HDS[None]))).astype(float)
    harm=np.concatenate([block(ks),block(ko)],1)
    return dials,items,gamb,harm
