// Host-only numerical oracle. No simulator objects, scheduling or resource APIs.
#include <sst/elements/golem/fp16.h>
#include <cstdint>
#include <iostream>
#include <vector>
using namespace SST::Golem;
int main() {
    std::ios::sync_with_stdio(false);
    uint32_t shape[3];
    if (!std::cin.read(reinterpret_cast<char*>(shape),sizeof(shape))) return 1;
    const uint32_t rows=shape[0],n=shape[1],k=shape[2];
    if (!rows || !n || !k || k%64) return 2;
    std::vector<uint16_t> x(uint64_t(rows)*k),w(uint64_t(n)*k),y(uint64_t(rows)*n);
    if (!std::cin.read(reinterpret_cast<char*>(x.data()),x.size()*2) ||
        !std::cin.read(reinterpret_cast<char*>(w.data()),w.size()*2)) return 3;
    // Decode through an exhaustive lookup table. Arithmetic and rounding order
    // follow the documented product -> dot -> K-tile accumulation boundaries.
    std::vector<float> decoded(65536);
    for (uint32_t bits=0;bits<65536;++bits) decoded[bits]=golem_fp16_to_float(bits);
    for (uint32_t r=0;r<rows;++r) for (uint32_t o=0;o<n;++o) {
        uint16_t total=0;
        for (uint32_t tile=0;tile<k;tile+=64) {
            uint16_t dot=0;
            for (uint32_t i=tile;i<tile+64;++i) {
                const uint16_t product=golem_float_to_fp16(decoded[x[uint64_t(r)*k+i]]*decoded[w[uint64_t(o)*k+i]]);
                dot=golem_float_to_fp16(decoded[dot]+decoded[product]);
            }
            total=golem_float_to_fp16(decoded[total]+decoded[dot]);
        }
        y[uint64_t(r)*n+o]=total;
    }
    std::cout.write(reinterpret_cast<const char*>(y.data()),y.size()*2);
    return std::cout?0:4;
}
