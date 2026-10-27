#ifndef GOLEM_TILE_UNIT_API_H
#define GOLEM_TILE_UNIT_API_H
#include <sst/elements/golem/array/computeArray.h>
namespace SST { namespace Golem {
// Keep existing timed GM/array-port and active-K interfaces available to TE,
// Attention and decoder clients. MV completion is still one event per TU.
class TileUnitAPI : public ComputeArray {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_DERIVED_API(
        SST::Golem::TileUnitAPI, SST::Golem::ComputeArray,
        TimeConverter*, Event::HandlerBase*)
    TileUnitAPI(ComponentId_t id,Params& params,TimeConverter* tc,Event::HandlerBase* handler)
        : ComputeArray(id,params,tc,handler) {}
    virtual void beginMV(uint32_t id) { beginComputation(id); }
};
}}
#endif
