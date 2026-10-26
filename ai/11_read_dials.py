"""Read a person's dials with the trained integrator.
Input row: 27 SD3 answers (1-5, reverse items already flipped), 128 gamble choices (1 accept, 0 reject)
in the fixed grid order from synth.py, 160 harm choices (1 = took the money and the shocks; first 80 self, last 80 other).
Missing sources can be zeros only if you use a model trained without them."""
import numpy as np, torch, sys

from train_net import Net
def read_dials(items,gamb,harm,weights="integrator_spread1.6.pt"):
    x=np.concatenate([(np.asarray(items)-3)/1.3,np.asarray(gamb)*2-1,np.asarray(harm)*2-1]).astype(np.float32)[None]
    net=Net(x.shape[1]); net.load_state_dict(torch.load(weights)); net.eval()
    with torch.no_grad(): mu,lv=net(torch.tensor(x))
    mu=mu[0].numpy(); sd=np.exp(lv[0].numpy()/2)
    return {d:(round(float(m),2),round(float(m-1.645*s),2),round(float(m+1.645*s),2)) for d,m,s in zip(["guilt","fear","reward"],mu,sd)}
if __name__=="__main__":
    from synth import people
    rng=np.random.default_rng(42); dials,items,gamb,harm=people(4000,rng,0.3)
    picks={"typical person":np.argmin(np.abs(dials).sum(1)),
           "low guilt + low fear":np.argmin(((dials[:,0]+2)**2+(dials[:,1]+2)**2+dials[:,2]**2)),
           "low guilt only":np.argmin(((dials[:,0]+2)**2+(dials[:,1])**2+dials[:,2]**2))}
    for name,i in picks.items():
        print(name,"| true:",np.round(dials[i],2),"| read:",read_dials(items[i],gamb[i],harm[i]))
