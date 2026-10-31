#ifndef SST_GOLEM_SFU_NORM_COMPENSATED_H
#define SST_GOLEM_SFU_NORM_COMPENSATED_H
#include <sst/elements/golem/sfu/arithmetic.h>
namespace SST { namespace Golem {
// Two binary16 words, never a widened accumulator. Each primitive rounds to
// binary16, including FMA (one rounding for the fused operation).
struct SFUNormPair { float hi=0, lo=0; };
inline SFUNormPair normSquare(const SFUArithmetic& a,float x) {
    float h=a.mul(x,x);return {h,a.fma(x,x,-h)};
}
inline SFUNormPair normAdd(const SFUArithmetic& a,SFUNormPair x,SFUNormPair y) {
    float s=a.add(x.hi,y.hi),v=a.sub(s,x.hi);
    float e=a.add(a.sub(x.hi,a.sub(s,v)),a.sub(y.hi,v));
    return {s,a.add(a.add(e,x.lo),y.lo)};
}
inline SFUNormPair normRoot(const SFUArithmetic& a,SFUNormPair m) {
    float r=a.sqrt(m.hi);
    return {r,a.div(a.add(a.fma(-r,r,m.hi),m.lo),a.mul(2.f,r))};
}
inline SFUNormPair normDivide(const SFUArithmetic& a,float x,SFUNormPair r) {
    float q=a.div(x,r.hi);
    return {q,a.div(a.fma(-q,r.lo,a.fma(-q,r.hi,x)),r.hi)};
}
inline float normGamma(const SFUArithmetic& a,SFUNormPair q,float gamma) {
    float z=a.mul(q.hi,gamma);
    return a.add(z,a.fma(q.lo,gamma,a.fma(q.hi,gamma,-z)));
}
} }
#endif
