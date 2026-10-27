#ifndef TILEMC_COMPUTE_UNIT_H
#define TILEMC_COMPUTE_UNIT_H

#include <cstddef>

namespace SST {
namespace Golem {

// VV: one CU reduces a matrix row against the TU's shared input vector.
// CU models are ordinary C++ objects: no clocks, links or SST events per CU.
template <typename T>
class ComputeUnit {
public:
    template<class U>
    T dotConverted(const U* row,const U* vector,std::size_t k) const {
        T result = 0;
        for (std::size_t column=0;column<k;++column)
            result += T(static_cast<float>(row[column])) * T(static_cast<float>(vector[column]));
        return result;
    }
    T dot(const T* row, const T* vector, std::size_t k) const {
        T result = 0;
        for (std::size_t column = 0; column < k; ++column) {
            result += row[column] * vector[column];
        }
        return result;
    }
};

} // namespace Golem
} // namespace SST

#endif
