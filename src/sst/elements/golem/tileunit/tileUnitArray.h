#ifndef GOLEM_TILE_UNIT_ARRAY_H
#define GOLEM_TILE_UNIT_ARRAY_H
#include <sst/elements/golem/tileunit/tileUnitArrayModel.h>
namespace SST { namespace Golem {
// Native TU bank owns execution independently of the legacy MVM adapters.
class TileUnitArray : public TileUnitArrayModel<float> {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(TileUnitArray,"golem","TileUnitArray",
        SST_ELI_ELEMENT_VERSION(1,0,0),"Native Tile Units with compatible array transport",
        SST::Golem::TileUnitArrayModel<float>)
    SST_ELI_DOCUMENT_PARAMS({"functionalCompute","Execute numerical MACs or timing only","1"})
    TileUnitArray(ComponentId_t id,Params& params,TimeConverter* tc,Event::HandlerBase* handler)
        : TileUnitArrayModel<float>(id,params,tc,handler) {}
};
}}
#endif
