"""Independent sfu-fp16-v2: all primitive inputs/results remain binary16.

v1 remains in sfu_fp16_reference.py for archived evidence. No production imports.
"""
from dataclasses import dataclass
import numpy as np
from sfu_fp16_reference import (half, add, sub, mul, exp, reciprocal, reduce_sum,
    rope, float_array_gemm, fp16_gemm, grouped_attention as grouped_v1)


def rmsnorm(x, gamma, reduction='tree', epsilon=1.e-6):
    x=half(x)
    maximum=np.max(np.abs(x),axis=-1,keepdims=True)
    # Power-of-two exponent selection is metadata, not a wider FP operation.
    scale=np.exp2(np.maximum(0,np.frexp(maximum)[1]-1))
    x=half(x/scale)
    mean=half(reduce_sum(mul(x,x),reduction)/half(x.shape[-1]))
    eps=half(half(half(epsilon)/scale[...,0])/scale[...,0])
    radius=half(np.sqrt(add(mean,eps)))
    return mul(half(x/radius[...,None]),gamma)


def silu(x):
    x=half(x);t=exp(-np.abs(x))
    numerator=np.where(x<0,mul(x,t),x)
    return mul(numerator,reciprocal(add(1.,t)))


@dataclass(frozen=True)
class ReferenceContract:
    version: str='sfu-fp16-v2'
    norm_reduction: str='tree'
    def __post_init__(self):
        if self.version!='sfu-fp16-v2' or self.norm_reduction not in ('tree','sequential'):
            raise ValueError('unsupported SFU numerical contract')
    def rms(self,x,gamma):return rmsnorm(x,gamma,self.norm_reduction).astype('<f2')
    def silu(self,x):return silu(x).astype('<f2')


def online_tile(scores,head_dim,old_m=None,old_l=None,*,final=False,key_begin=0,row_begin=0,causal=False,exponential=exp):
    scores=np.asarray(scores,dtype='f8');rows,cols=scores.shape
    scaled=mul(scores,half(1/np.sqrt(half(head_dim))))
    if causal:
        scaled=np.where(np.arange(cols)[None,:]+key_begin>np.arange(rows)[:,None]+row_begin,-np.inf,scaled)
    maximum=np.max(scaled,axis=1)
    if not np.isfinite(maximum).all():raise ValueError('fully masked/nonfinite tile row')
    old_m=np.full(rows,-np.inf) if old_m is None else half(old_m)
    old_l=np.zeros(rows) if old_l is None else half(old_l)
    new_m=np.maximum(old_m,maximum)
    weights=exponential(sub(scaled,new_m[:,None]))
    # Physical lane reductions followed by one modeled final partial reduction.
    partial=np.stack([reduce_sum(weights[:,i:i+16],'tree') for i in range(0,cols,16)],axis=1)
    tile_sum=reduce_sum(partial,'tree')
    alpha=np.where(np.isneginf(old_m),0.,exponential(sub(old_m,new_m)))
    beta=np.ones(rows);new_l=add(mul(old_l,alpha),tile_sum)
    if not np.isfinite(new_l).all() or np.any(new_l<=0):raise ValueError('online normalizer overflow/invalid')
    normalization=reciprocal(new_l) if final else np.ones(rows)
    return dict(scores=scores,scaled=scaled,maximum=maximum,weights=weights,tile_sum=tile_sum,
                m=new_m,l=new_l,alpha=alpha,beta=beta,normalization=normalization,
                old_scale=mul(alpha,normalization),weight_scale=normalization,
                p=mul(weights,normalization[:,None]))


def grouped_attention(q,k,v,*,window=4,tile=64):
    return grouped_v1(q,k,v,window=window,tile=tile,tile_reference=online_tile)
