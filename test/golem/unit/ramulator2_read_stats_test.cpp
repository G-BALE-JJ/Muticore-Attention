#include <sst/elements/memHierarchy/membackend/ramulator2ReadCommandStats.h>
#include <cassert>

int main() {
    using SST::MemHierarchy::Ramulator2ReadCommandStats;
    Ramulator2ReadCommandStats stats;
    stats.recordRead(0, 0, 0, 2, 4, false);
    stats.recordRead(1, 1, 2, 2, 4, true);
    stats.recordRead(4, 0, 0, 2, 4, true); // L despite intervening PC1.
    stats.recordRead(5, 1, 3, 2, 4, true); // S despite intervening PC0.
    stats.recordRead(9, 1, 3, 2, 4, true);
    stats.recordRead(13, 0, 1, 2, 4, false); // A WR separates global RD columns.
    assert(stats.longPairs.count == 2 && stats.longPairs.gapSum == 8);
    assert(stats.shortPairs.count == 2 && stats.shortPairs.gapSum == 13);
    assert(stats.longPairs.minGap == 4 && stats.shortPairs.minGap == 4);
    assert(stats.longPairs.violations == 0 && stats.shortPairs.violations == 0);
    assert(stats.pcReadSequences == 2);
    assert(stats.longPairs.count + stats.shortPairs.count == 6 - stats.pcReadSequences);
    assert(stats.globalReadPairs == 4 && stats.crossPcPairs.count == 3);
    assert(stats.crossPcPairs.gapSum == 5 && stats.crossPcPairs.minGap == 1);

    Ramulator2ReadCommandStats otherChannel;
    otherChannel.recordRead(20, 0, 0, 2, 4, true);
    assert(otherChannel.longPairs.count == 0 && otherChannel.globalReadPairs == 0);

    Ramulator2ReadCommandStats invalidTrace;
    invalidTrace.recordRead(0, 0, 0, 2, 4, false);
    invalidTrace.recordRead(1, 1, 0, 2, 4, true);
    invalidTrace.recordRead(2, 0, 0, 2, 4, true);
    invalidTrace.recordRead(3, 0, 1, 2, 4, true);
    assert(invalidTrace.longPairs.violations == 1);
    assert(invalidTrace.shortPairs.violations == 1);
}
