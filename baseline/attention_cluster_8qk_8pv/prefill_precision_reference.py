"""FP32 model arithmetic for measuring drift, independent of array rounding."""
import numpy as np


def rms(x,gamma):
    x=np.asarray(x,dtype=np.float32)
    return x*(1/np.sqrt(np.mean(x*x,axis=1,dtype=np.float32)+np.float32(1e-6)))[:,None]*gamma.astype(np.float32)


def layer(x,weights,gamma1,gamma2,hq,hkv,table):
    sequence=len(x)
    p=(rms(x,gamma1)@weights['qkv'].astype(np.float32).T).reshape(sequence,hq+2*hkv,64)
    table=np.asarray(table,dtype=np.float32).reshape(-1,64)[:sequence]
    def rotate(values):
        out=np.empty_like(values)
        out[:,0::2]=values[:,0::2]*table[:,0::2]-values[:,1::2]*table[:,1::2]
        out[:,1::2]=values[:,0::2]*table[:,1::2]+values[:,1::2]*table[:,0::2]
        return out
    attention=np.empty((sequence,hq,64),dtype=np.float32)
    qpositions=np.arange(sequence)
    for head in range(hq):
        kv=head//(hq//hkv)
        q=rotate(p[:,head]);k=rotate(p[:,hq+kv]);v=p[:,hq+hkv+kv]
        for first in range(0,sequence,64):
            last=min(first+64,sequence)
            score=(q[first:last]@k.T)*np.float32(1/8)
            score[qpositions[None,:]>qpositions[first:last,None]]=-np.inf
            score-=score.max(axis=1,keepdims=True)
            probability=np.exp(score);probability/=probability.sum(axis=1,keepdims=True)
            attention[first:last,head]=probability@v
    h1=x+attention.reshape(sequence,-1)@weights['wo'].astype(np.float32).T
    norm=rms(h1,gamma2)
    gate=norm@weights['gate'].astype(np.float32).T
    up=norm@weights['up'].astype(np.float32).T
    return h1+(gate/(np.float32(1)+np.exp(-gate))*up)@weights['down'].astype(np.float32).T


def error(got,want):
    got=np.asarray(got,dtype=np.float64);want=np.asarray(want,dtype=np.float64)
    if got.shape!=want.shape or not np.isfinite(got).all() or not np.isfinite(want).all():
        raise ValueError('nonfinite or inconsistent precision reference')
    diff=got-want
    return {'status':'MEASURED','max_abs_error':float(np.max(np.abs(diff))),
            'rms_error':float(np.sqrt(np.mean(diff*diff))),
            'relative_l2_error':float(np.linalg.norm(diff)/max(np.linalg.norm(want),1e-30)),
            'note':'FP32 quality metric; functional PASS uses the explicit hardware rounding oracle'}
