import numpy as np, torch, torch.nn as nn, json, time, sys
RHO=float(sys.argv[1])
from synth import people, N_ITEMS, N_GAMB, N_HARM
torch.manual_seed(0); rng=np.random.default_rng(0)
NTR,NVA,NTE=60000,8000,8000
dials,items,gamb,harm=people(NTR+NVA+NTE,rng,RHO)
X=np.concatenate([(items-3)/1.3,gamb*2-1,harm*2-1],1).astype(np.float32); Y=dials.astype(np.float32)
SRC={"questionnaire":slice(0,27),"gambles":slice(27,27+N_GAMB),"harm task":slice(27+N_GAMB,27+N_GAMB+N_HARM)}
class Net(nn.Module):
    def __init__(s,d):
        super().__init__(); s.b=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU(),nn.Linear(128,6))
    def forward(s,x): o=s.b(x); return o[:,:3],o[:,3:].clamp(-7,3)
def train(mask,epochs=40):
    Xm=X*mask; tr=torch.tensor(Xm[:NTR]); ytr=torch.tensor(Y[:NTR]); va=torch.tensor(Xm[NTR:NTR+NVA]); yva=torch.tensor(Y[NTR:NTR+NVA])
    net=Net(X.shape[1]); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4); best=(1e9,None)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(NTR)
        for i in range(0,NTR,256):
            b=perm[i:i+256]; mu,lv=net(tr[b]); loss=(0.5*(lv+(ytr[b]-mu)**2/lv.exp())).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): mu,lv=net(va); vl=(0.5*(lv+(yva-mu)**2/lv.exp())).mean().item()
        if vl<best[0]: best=(vl,{k:v.clone() for k,v in net.state_dict().items()})
    net.load_state_dict(best[1]); return net
def evaluate(net,mask):
    te=torch.tensor(X[NTR+NVA:]*mask); y=Y[NTR+NVA:]
    with torch.no_grad(): mu,lv=net(te)
    mu=mu.numpy(); sd=np.exp(lv.numpy()/2)
    r=[np.corrcoef(mu[:,i],y[:,i])[0,1] for i in range(3)]
    cover=[np.mean(np.abs(y[:,i]-mu[:,i])<1.645*sd[:,i]) for i in range(3)]
    width=[np.mean(2*1.645*sd[:,i]) for i in range(3)]
    return r,cover,width
configs={"questionnaire only":["questionnaire"],"tasks only":["gambles","harm task"],"everything":["questionnaire","gambles","harm task"]}
res={}
for name,srcs in configs.items():
    mask=np.zeros(X.shape[1],np.float32)
    for s in srcs: mask[SRC[s]]=1
    t=time.time(); net=train(mask); r,cov,w=evaluate(net,mask); res[name]=dict(r=[float(v) for v in r],cover=[float(v) for v in cov],width=[float(v) for v in w])
    print(f"{name:20s} r guilt/fear/reward = {np.round(r,2)}  90% interval coverage = {np.round(cov,2)}  avg width = {np.round(w,2)}  ({time.time()-t:.0f}s)")
    if name=="everything": torch.save(net.state_dict(),f"dials_integrator_rho{RHO}.pt")
json.dump(res,open(f"training_results_rho{RHO}.json","w"),indent=1)
