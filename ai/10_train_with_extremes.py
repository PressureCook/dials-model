import numpy as np, torch, json, sys
from synth import people
from train_net import Net
torch.manual_seed(1); RHO=0.3
def make(n,rng,spread):
    d,i,g,h=people(n,rng,RHO,spread); X=np.concatenate([(i-3)/1.3,g*2-1,h*2-1],1).astype(np.float32); return X,d.astype(np.float32)
def fit(X,Y,epochs=40):
    n=int(len(X)*.9); tr,ytr,va,yva=map(torch.tensor,(X[:n],Y[:n],X[n:],Y[n:]))
    net=Net(X.shape[1]); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4); best=(1e9,None)
    for ep in range(epochs):
        net.train(); p=torch.randperm(n)
        for k in range(0,n,256):
            b=p[k:k+256]; mu,lv=net(tr[b]); loss=(0.5*(lv+(ytr[b]-mu)**2/lv.exp())).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): mu,lv=net(va); vl=(0.5*(lv+(yva-mu)**2/lv.exp())).mean().item()
        if vl<best[0]: best=(vl,{a:b.clone() for a,b in net.state_dict().items()})
    net.load_state_dict(best[1]); return net
rng=np.random.default_rng(7)
# test sets: realistic population, and the 'corner' (guilt < -1.5 and fear < -1.5)
Xp,Yp=make(10000,rng,1.0)
Xc,Yc=make(200000,rng,1.6); m=(Yc[:,0]<-1.5)&(Yc[:,1]<-1.5); Xc,Yc=Xc[m][:3000],Yc[m][:3000]
print("corner test people:",len(Yc))
out={}
for name,spread in [("trained on realistic population",1.0),("trained with extremes oversampled",1.6)]:
    X,Y=make(68000,np.random.default_rng(11),spread); net=fit(X,Y)
    res={}
    for tname,Xt,Yt in [("population",Xp,Yp),("corner",Xc,Yc)]:
        with torch.no_grad(): mu,lv=net(torch.tensor(Xt))
        mu=mu.numpy(); sd=np.exp(lv.numpy()/2)
        mae=np.abs(mu-Yt).mean(0); cov=(np.abs(Yt-mu)<1.645*sd).mean(0); bias=(mu-Yt).mean(0)
        res[tname]=dict(mae=mae.round(2).tolist(),coverage=cov.round(2).tolist(),bias=bias.round(2).tolist())
        print(f"{name:36s} {tname:10s} error g/f/r={mae.round(2)} bias={bias.round(2)} 90% coverage={cov.round(2)}")
    out[name]=res
    torch.save(net.state_dict(),f"integrator_spread{spread}.pt")
json.dump(out,open("tail_results.json","w"),indent=1)
