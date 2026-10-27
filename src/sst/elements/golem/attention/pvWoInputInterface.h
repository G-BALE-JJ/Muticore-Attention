#ifndef SST_GOLEM_PV_WO_INPUT_INTERFACE_H
#define SST_GOLEM_PV_WO_INPUT_INTERFACE_H
#include <cstdint>
namespace SST { namespace Golem {
class PvWoInputInterface {
public:
    virtual ~PvWoInputInterface() = default;
    virtual bool consume(uint32_t epoch, uint32_t manager, uint32_t lane) = 0;
    // Direct consumers retain the timed landing until the last read ACK.
    virtual bool acquire(uint32_t epoch, uint32_t manager, uint32_t lane) {
        return consume(epoch, manager, lane);
    }
    virtual bool release(uint32_t, uint32_t, uint32_t) { return true; }
};
}}
#endif
