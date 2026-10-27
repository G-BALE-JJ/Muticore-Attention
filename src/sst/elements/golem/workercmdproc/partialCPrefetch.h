#ifndef GOLEM_PARTIAL_C_PREFETCH_H
#define GOLEM_PARTIAL_C_PREFETCH_H
#include <cstddef>
#include <cstdint>
#include <vector>
namespace SST { namespace Golem {
// Next-window tile zero must observe this window's saved partial, even when
// tile zero is the last tile scheduled by the readiness-based reuse order.
inline bool firstPartialReadyForNextWindow(const std::vector<uint8_t>& done,
                                            size_t current) {
    if (done.empty() || !done[0]) return false;
    for (size_t index = 0; index < done.size(); ++index)
        if (index != current && !done[index]) return false;
    return true;
}
}}
#endif
