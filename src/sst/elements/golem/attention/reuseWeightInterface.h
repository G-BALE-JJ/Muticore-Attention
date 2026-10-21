#ifndef SST_GOLEM_REUSE_WEIGHT_INTERFACE_H
#define SST_GOLEM_REUSE_WEIGHT_INTERFACE_H
#include <cstdint>
#include <functional>
namespace SST { namespace Golem {
// Software-only transport service; fetch completes after the destination GM ACK.
struct ReuseWeightInterface {
    virtual ~ReuseWeightInterface()=default;
    virtual void begin(uint32_t epoch)=0;
    virtual void fetch(uint32_t epoch,uint64_t source,uint64_t destination,std::function<void(bool)> done)=0;
    virtual bool release(uint32_t epoch)=0;
    virtual uint64_t hbmBytes(uint32_t epoch) const=0;
    virtual uint64_t localReadBytes(uint32_t epoch) const=0;
    virtual uint64_t localWriteBytes(uint32_t epoch) const=0;
    virtual void trace(uint32_t epoch) const=0;
};
}}
#endif
