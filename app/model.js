// Dials integrator forward pass (315 -> 256 -> 128 -> 6), weights from the trained PyTorch model.
function erf(x){const s=Math.sign(x);x=Math.abs(x);const t=1/(1+0.3275911*x);
  const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);return s*y;}
function gelu(x){return 0.5*x*(1+erf(x/Math.SQRT2));}
function loadWeights(b64,shapes){const bin=atob(b64);const buf=new ArrayBuffer(bin.length);const u=new Uint8Array(buf);
  for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);const f=new Float32Array(buf);const out=[];let o=0;
  for(const s of shapes){const n=s.reduce((a,b)=>a*b,1);out.push({shape:s,v:f.subarray(o,o+n)});o+=n;}return out;}
function linear(x,W,b,act){const [o,i]=W.shape;const y=new Float32Array(o);
  for(let r=0;r<o;r++){let s=b.v[r];const off=r*i;for(let c=0;c<i;c++)s+=W.v[off+c]*x[c];y[r]=act?gelu(s):s;}return y;}
function runModel(L,x){const h1=linear(x,L[0],L[1],true),h2=linear(h1,L[2],L[3],true),o=linear(h2,L[4],L[5],false);
  const mu=[o[0],o[1],o[2]],lv=[3,4,5].map(i=>Math.min(3,Math.max(-7,o[i])));return {mu,sd:lv.map(v=>Math.exp(v/2))};}
if(typeof module!=="undefined")module.exports={loadWeights,runModel};
