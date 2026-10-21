#ifndef SST_GOLEM_SOFTWARE_LAYER_RESIDENCY_H
#define SST_GOLEM_SOFTWARE_LAYER_RESIDENCY_H
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
namespace SST { namespace Golem {
// Software ownership record in the unused gap after the existing descriptor.
// This is ordinary GM memory: publishers and consumers must use timed ports.
// Payload reuses the existing Wo/H1 arena; no hardware storage is introduced.
struct SoftwareLayerResidency {
    static constexpr uint64_t record=0x3400, payload=0xb0000;
    static constexpr uint64_t magic=0x594c415945523031ull;
    static std::vector<uint8_t> encode(uint64_t epoch,uint64_t address,uint32_t rows,
                                       uint32_t hidden,uint32_t lane,uint32_t manager) {
        const std::array<uint64_t,8> fields{{magic,epoch,address,rows,hidden,lane,manager,
                                           uint64_t(std::min(64u,rows-lane*64))*hidden*2}};
        std::vector<uint8_t> b(64);std::memcpy(b.data(),fields.data(),64);return b;
    }
    static bool matches(const std::vector<uint8_t>& bytes,uint64_t nextEpoch,uint64_t address,
                        uint32_t rows,uint32_t hidden,uint32_t lane,uint32_t manager) {
        if(bytes.size()!=64 || !nextEpoch || rows<=lane*64)return false;
        std::array<uint64_t,8> f{};std::memcpy(f.data(),bytes.data(),64);
        return f[0]==magic && f[1]+1==nextEpoch && f[2]==address && f[3]==rows &&
               f[4]==hidden && f[5]==lane && f[6]==manager &&
               f[7]==uint64_t(std::min(64u,rows-lane*64))*hidden*2;
    }
};
}}
#endif
