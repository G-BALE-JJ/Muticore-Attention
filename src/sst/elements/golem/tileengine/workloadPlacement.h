#ifndef GOLEM_WORKLOAD_PLACEMENT_H
#define GOLEM_WORKLOAD_PLACEMENT_H
#include <algorithm>
#include <cstdint>
#include <vector>
namespace SST { namespace Golem {
// Explicit native-runtime contract. Each independent workload owns disjoint
// existing worker slots and HBM nodes; this does not add a buffer or compute unit.
struct WorkloadPlacement {
    std::vector<uint32_t> dataNodes, taskNodes, aNodes, bNodes;
    bool replicateB = false;
    bool valid(uint32_t physicalNodes,uint32_t mTiles,uint32_t nTiles,
               uint32_t macroTasks) const {
        if(dataNodes.empty() || taskNodes.size()!=macroTasks ||
           aNodes.size()!=mTiles || bNodes.size()!=nTiles)return false;
        auto sorted=dataNodes;std::sort(sorted.begin(),sorted.end());
        if(sorted.front()==0 || sorted.back()>physicalNodes ||
           std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())return false;
        for(const auto* mapping:{&taskNodes,&aNodes,&bNodes})
            for(auto node:*mapping)
                if(!std::binary_search(sorted.begin(),sorted.end(),node))return false;
        return true;
    }
    uint64_t aSlot(uint32_t tile) const {
        return std::count(aNodes.begin(),aNodes.begin()+tile,aNodes.at(tile));
    }
    uint64_t bSlot(uint32_t tile) const {
        return replicateB ? tile : std::count(bNodes.begin(),bNodes.begin()+tile,bNodes.at(tile));
    }
    uint64_t aAddress(uint32_t tile,uint32_t k,uint32_t kTiles,
                      uint64_t nodeSize,uint64_t offset,uint64_t stride) const {
        return uint64_t(aNodes.at(tile))*nodeSize+offset+(aSlot(tile)*kTiles+k)*stride;
    }
    uint64_t bAddress(uint32_t tile,uint32_t k,uint32_t kTiles,uint32_t cNode,
                      uint64_t nodeSize,uint64_t offset,uint64_t stride) const {
        return uint64_t(replicateB?cNode:bNodes.at(tile))*nodeSize+offset+(bSlot(tile)*kTiles+k)*stride;
    }
};
}}
#endif
