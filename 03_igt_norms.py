import sys, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
d=pd.read_excel(sys.argv[1])
d=d[d["IGT Version"]=="Original"]
# controls are often shared across samples in a study; dedupe by author+control values
c=d.drop_duplicates(subset=["Author","nControl","TotalControl"])
rows=[]
for b in range(1,6):
    m=c[f"Block{b}Control"]; s=c[f"BlockSD{b}Control"]; n=c["nControl"]
    ok=m.notna()&s.notna()
    w=n[ok]; mean=(m[ok]*w).sum()/w.sum()
    pooled_sd=np.sqrt((((w-1)*s[ok]**2).sum())/((w-1).sum()))
    # clinical
    mc=d[f"Block{b}"]; sc=d[f"BlockSD{b}"]; nc=d["n"]; okc=mc.notna()
    rows.append(dict(block=b,ctrl_mean=round(mean,2),ctrl_sd=round(pooled_sd,2),k=int(ok.sum()),n=int(w.sum()),
                     abi_mean=round((mc[okc]*nc[okc]).sum()/nc[okc].sum(),2)))
r=pd.DataFrame(rows); print(r.to_string(index=False)); r.to_csv("igt_norms.csv",index=False)
