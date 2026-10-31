"""Independent NumPy numerical contract, sfu-fp16-v1 (no production imports).

Every arithmetic primitive rounds its operands and result to binary16, RNE,
with gradual underflow. Float64 is only a carrier for exact half operands.
Attention sum order is sequential; Norm reduction order is explicitly selected.
No overflow repair or FP32 fallback is part of v1.
"""
from dataclasses import dataclass
import numpy as np


def half(value):
    with np.errstate(over='ignore', invalid='ignore'):
        return np.asarray(value, dtype=np.float16).astype(np.float64)


def add(a, b):
    return half(half(a) + half(b))


def mul(a, b):
    return half(half(a) * half(b))


def sub(a, b):
    return half(half(a) - half(b))


def exp(x):
    with np.errstate(over='ignore', invalid='ignore'):
        return half(np.exp(half(x)))


def reciprocal(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return half(1 / half(x))


def reduce_sum(x, order='sequential'):
    """Reduce the last dimension; preserve odd tree tails without padding."""
    x = half(x)
    if not x.shape[-1]:
        raise ValueError('empty reduction')
    if order == 'sequential':
        total = np.zeros(x.shape[:-1])
        for col in range(x.shape[-1]):
            total = add(total, x[..., col])
        return total
    if order != 'tree':
        raise ValueError(f'unsupported reduction: {order}')
    while x.shape[-1] > 1:
        pairs = x.shape[-1] // 2
        combined = add(x[..., :2*pairs:2], x[..., 1:2*pairs:2])
        x = np.concatenate((combined, x[..., -1:]), axis=-1) if x.shape[-1] % 2 else combined
    return x[..., 0]


def rmsnorm(x, gamma, reduction='tree', epsilon=1.e-6):
    x = half(x)
    total = reduce_sum(mul(x, x), reduction)
    mean = half(total / half(x.shape[-1]))
    with np.errstate(divide='ignore', invalid='ignore'):
        inverse = half(1 / np.sqrt(add(mean, epsilon)))
    return mul(mul(x, inverse[..., None]), gamma)


def silu(x):
    x = half(x)
    return mul(x, reciprocal(add(1., exp(-x))))


def rope(x, table):
    x, table = half(x), half(table)
    table = table.reshape(-1, x.shape[-1])[:x.shape[-2]]
    out = np.empty_like(x)
    out[..., 0::2] = sub(mul(x[..., 0::2], table[:, 0::2]), mul(x[..., 1::2], table[:, 1::2]))
    out[..., 1::2] = add(mul(x[..., 0::2], table[:, 1::2]), mul(x[..., 1::2], table[:, 0::2]))
    return out


@dataclass(frozen=True)
class ReferenceContract:
    version: str = 'sfu-fp16-v1'
    norm_reduction: str = 'tree'

    def __post_init__(self):
        if self.version != 'sfu-fp16-v1' or self.norm_reduction not in ('tree', 'sequential'):
            raise ValueError('unsupported SFU numerical contract')

    def rms(self, x, gamma):
        return rmsnorm(x, gamma, self.norm_reduction).astype('<f2')

    def silu(self, x):
        return silu(x).astype('<f2')


def online_tile(scores, head_dim, old_m=None, old_l=None, *, final=False,
                key_begin=0, row_begin=0, causal=False):
    """One local Attention tile; return all observable numeric boundaries.

    Fully masked rows are rejected, matching the current unsupported v1 case;
    callers must omit future causal tiles, rather than fabricate a normalizer.
    """
    scores = np.asarray(scores, dtype=np.float64)
    rows, cols = scores.shape
    scaled = mul(scores, half(1 / np.sqrt(half(head_dim))))
    if causal:
        mask = np.arange(cols)[None, :] + key_begin > np.arange(rows)[:, None] + row_begin
        scaled = np.where(mask, -np.inf, scaled)
    maximum = np.max(scaled, axis=-1)
    if not np.isfinite(maximum).all():
        raise ValueError('v1 does not support a fully masked/nonfinite tile row')
    weights = exp(sub(scaled, maximum[:, None]))
    tile_sum = reduce_sum(weights)
    old_m = np.full(rows, -np.inf) if old_m is None else half(old_m)
    old_l = np.zeros(rows) if old_l is None else half(old_l)
    new_m = np.maximum(old_m, maximum)
    alpha = np.where(np.isneginf(old_m), 0., exp(sub(old_m, new_m)))
    beta = exp(sub(maximum, new_m))
    new_l = add(mul(old_l, alpha), mul(tile_sum, beta))
    if not np.isfinite(new_l).all() or np.any(new_l <= 0):
        raise ValueError('v1 online normalizer overflow/invalid')
    normalization = reciprocal(new_l) if final else half(np.ones(rows))
    old_scale = mul(alpha, normalization)
    weight_scale = mul(beta, normalization)
    return dict(scores=scores, scaled=scaled, maximum=maximum, weights=weights,
                tile_sum=tile_sum, m=new_m, l=new_l, alpha=alpha, beta=beta,
                normalization=normalization, old_scale=old_scale,
                weight_scale=weight_scale, p=mul(weights, weight_scale[:, None]))


def fp16_gemm(a, b, initial=None, tile=64):
    """Independent array model: half products, ordered half dot, tiled C add."""
    a, b = half(a), half(b)
    if a.shape[1] != b.shape[0]:
        raise ValueError('GEMM shape mismatch')
    total = np.zeros((a.shape[0], b.shape[1])) if initial is None else half(initial)
    for begin in range(0, a.shape[1], tile):
        dot = np.zeros_like(total)
        for k in range(begin, min(begin+tile, a.shape[1])):
            dot = add(dot, mul(a[:, k, None], b[k, None, :]))
        total = add(total, dot)
    return total


def float_array_gemm(a, b, tile=64):
    """MVMFloatArray contract used by the existing Attention integration.

    FP16 tensor storage does not imply FP16 array accumulation. Projection
    jobs select their separate FP16 context; Attention uses float dot/C here.
    """
    a, b = np.asarray(a, dtype='f4'), np.asarray(b, dtype='f4')
    total = np.zeros((len(a), b.shape[1]), dtype='f4')
    for begin in range(0, a.shape[1], tile):
        dot = np.zeros_like(total)
        for k in range(begin, min(begin+tile, a.shape[1])):
            dot = np.add(dot, np.multiply(a[:, k, None], b[k, None, :], dtype='f4'), dtype='f4')
        total = np.add(total, dot, dtype='f4')
    return total


def grouped_attention(q, k, v, *, window=4, tile=64, tile_reference=online_tile):
    """Current 8+8 cluster contract: float arrays and full-FP16 SFU.

    Prior P rows are scaled in FP32 before each four-tile PV window. Previous
    window O is scaled once by the FP32 product of the four half scales.
    Returns final FP16 tensor plus independent per-tile SFU checkpoints.
    """
    scores = float_array_gemm(q, np.asarray(k).T)
    state, history, p_rows = None, [], []
    output = np.zeros((len(q), v.shape[1]), dtype='f4')
    group_scale = np.ones(len(q), dtype='f4')
    for begin in range(0, len(k), tile):
        end = min(begin+tile, len(k))
        state = tile_reference(scores[:, begin:end], q.shape[1],
                            None if state is None else state['m'],
                            None if state is None else state['l'], final=end == len(k))
        scale = state['old_scale'].astype('f4')
        p_rows = [np.multiply(p, scale[:, None], dtype='f4') for p in p_rows]
        p_rows.append(state['p'].astype('f4'))
        group_scale = np.multiply(group_scale, scale, dtype='f4')
        history.append(state)
        if len(p_rows) == window or end == len(k):
            width = sum(p.shape[1] for p in p_rows)
            packed = np.concatenate(p_rows, axis=1)
            packed = half(np.where(np.abs(packed) < 2.**-24,
                                   np.copysign(np.float32(0), packed), packed))
            pv = float_array_gemm(packed, v[end-width:end])
            output = np.add(np.multiply(output, group_scale[:, None], dtype='f4'), pv, dtype='f4')
            p_rows, group_scale = [], np.ones(len(q), dtype='f4')
    # Legacy tensor conversion flushes values strictly below the least half
    # subnormal before RNE (distinct from SFU intermediate gradual underflow).
    output = np.where(np.abs(output) < 2.**-24, np.copysign(np.float32(0), output), output)
    return half(output), history


def attention_from_scores(scores, v, head_dim, *, tile=64):
    """Sequential PV schedule, for paths with one PV update per score tile.

    Scheduling variants that group P rescaling before PV must model that order
    separately. This API never silently assumes grouped and sequential agree.
    """
    output = np.zeros((scores.shape[0], v.shape[1]))
    state = None
    history = []
    for begin in range(0, scores.shape[1], tile):
        end = min(begin+tile, scores.shape[1])
        state = online_tile(scores[:, begin:end], head_dim,
                            None if state is None else state['m'],
                            None if state is None else state['l'], final=end == scores.shape[1])
        output = fp16_gemm(state['p'], v[begin:end], mul(output, state['old_scale'][:, None]))
        history.append(state)
    return output, history
