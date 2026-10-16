#include "localGmBankScheduler.h"
#include <cassert>
#include <iostream>

using SST::Golem::LocalGmBankScheduler;

int main()
{
    LocalGmBankScheduler s({8, 128, 1, 2, 16});
    assert(s.enqueue(0, {1, 0, 256, 0}));
    assert(s.enqueue(1, {2, 0, 128, 0}));
    auto g1 = s.step(1);
    assert(g1.size() == 1 && g1[0].id == 1 && !g1[0].last);
    auto g2 = s.step(2);
    assert(g2.size() == 2 && g2[0].id == 1 && g2[1].id == 2);
    assert(s.stats().bankConflictCycles == 1);

    LocalGmBankScheduler different({8, 128, 1, 2, 16});
    assert(different.enqueue(0, {3, 128, 128, 0}));
    assert(different.enqueue(1, {4, 256, 128, 0}));
    auto parallel = different.step(1);
    assert(parallel.size() == 2 && parallel[0].bank != parallel[1].bank);
    bool monotonic_guard = false;
    try { different.step(1); } catch (const std::logic_error&) { monotonic_guard = true; }
    assert(monotonic_guard);
    std::cout << "PASS local GM bank scheduler" << std::endl;
}
