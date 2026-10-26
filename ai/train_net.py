import torch, torch.nn as nn
class Net(nn.Module):
    def __init__(s,d):
        super().__init__(); s.b=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU(),nn.Linear(128,6))
    def forward(s,x): o=s.b(x); return o[:,:3],o[:,3:].clamp(-7,3)
