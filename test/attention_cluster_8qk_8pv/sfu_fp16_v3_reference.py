"""Independent all-binary16 compensated RMSNorm; v2 Attention/SiLU unchanged."""
from dataclasses import dataclass
import numpy as np
from sfu_fp16_v2_reference import half,add,mul,silu,grouped_attention,online_tile,rope,float_array_gemm,fp16_gemm,reduce_sum
from sfu_fp16_v2_reference import rmsnorm as legacy_rmsnorm

def silu_lut(x):
    x=half(x); xf=x.astype(np.float32); return half(xf/(np.float32(1.)+np.exp(-xf)))

def fma(a,b,c):
    return half(half(a)*half(b)+half(c))

def pair_add(a,b,al,bl):
    s=add(a,b);v=half(s-a)
    e=add(half(a-half(s-v)),half(b-v))
    return s,add(add(e,al),bl)

def rmsnorm(x,gamma,reduction='tree',epsilon=1.e-6):
    if reduction=='sequential':return legacy_rmsnorm(x,gamma,reduction,epsilon)
    x=half(x);scale=np.exp2(np.maximum(0,np.frexp(np.max(abs(x),axis=-1,keepdims=True))[1]-1))
    x=half(x/scale);hi=mul(x,x);lo=fma(x,x,-hi)
    while hi.shape[-1]>1:
        if hi.shape[-1]%2:
            padding=[(0,0)]*hi.ndim;padding[-1]=(0,1)
            hi=np.pad(hi,padding);lo=np.pad(lo,padding)
        hi,lo=pair_add(hi[...,::2],hi[...,1::2],lo[...,::2],lo[...,1::2])
    mh=half(hi/half(x.shape[-1]));ml=half(lo/half(x.shape[-1]));eps=half(half(half(epsilon)/scale)/scale)
    mh,ml=pair_add(mh,eps,ml,0.)
    r=half(np.sqrt(mh));dr=half(add(fma(-r,r,mh),ml)/mul(2.,r))
    q=half(x/r);dq=half(fma(-q,dr,fma(-q,r,x))/r)
    z=mul(q,gamma)
    return add(z,fma(dq,gamma,fma(q,gamma,-z)))

@dataclass(frozen=True)
class ReferenceContract:
    version:str='sfu-fp16-v3'
    norm_reduction:str='tree'
    def __post_init__(self):
        if self.version!='sfu-fp16-v3' or self.norm_reduction not in ('tree','sequential'):raise ValueError('unsupported SFU numerical contract')
    def rms(self,x,gamma):return rmsnorm(x,gamma,self.norm_reduction).astype('<f2')
    def silu(self,x):return silu(x).astype('<f2')
