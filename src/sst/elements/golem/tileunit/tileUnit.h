#ifndef GOLEM_TILE_UNIT_H
#define GOLEM_TILE_UNIT_H
#include <sst/elements/golem/computeunit/computeUnit.h>
#include <sst/elements/golem/fp16.h>
#include <vector>
namespace SST { namespace Golem {
// One CU per output row. Operand ownership stays in the existing TU bank;
// these views add no matrix/vector copy, hardware buffer, event or clock.
template<class T> class TileUnit {
public:
    explicit TileUnit(size_t rows): cus_(rows), projectionCus_(rows) {}
    T dot(size_t row,const T* matrix,const T* input,size_t k) const {
        return cus_.at(row).dot(matrix,input,k);
    }
    GolemFp16 projectionDot(size_t row,const T* matrix,const T* input,size_t k) const {
        // Preserve the existing prefill FP16 multiply/add rounding exactly.
        return projectionCus_.at(row).dotConverted(matrix,input,k);
    }
private:
    std::vector<ComputeUnit<T>> cus_;
    std::vector<ComputeUnit<GolemFp16>> projectionCus_;
};
}}
#endif
