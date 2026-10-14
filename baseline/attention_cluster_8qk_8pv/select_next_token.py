"""Deterministic host greedy selection from the valid vocabulary logits."""
import time
import numpy as np


def select_next_token(logits,vocab):
    start=time.perf_counter_ns()
    values=np.asarray(logits)
    if values.ndim!=1 or not isinstance(vocab,(int,np.integer)) or vocab<=0 or vocab>len(values):
        raise ValueError('invalid logits shape/vocabulary')
    valid=values[:vocab]
    if not np.isfinite(valid).all():raise ValueError('nonfinite valid vocabulary logits')
    token=int(np.argmax(valid))
    return {'method':'greedy_argmax','token_id':token,'logit':float(valid[token]),'vocab':int(vocab),
        'tie_break':'lowest token ID','host_selection_seconds':(time.perf_counter_ns()-start)/1e9}
