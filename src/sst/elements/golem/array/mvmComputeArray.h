// Legacy MVM name adapts the native TU implementation; no duplicate banks.
#ifndef _MVMCOMPUTEARRAY_H
#define _MVMCOMPUTEARRAY_H
#include <sst/elements/golem/tileunit/tileUnitArrayModel.h>
namespace SST { namespace Golem {
template<class T> class MVMComputeArray : public TileUnitArrayModel<T> {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_DERIVED_API(SST::Golem::MVMComputeArray<T>,
        SST::Golem::TileUnitAPI, TimeConverter*, Event::HandlerBase*)
    MVMComputeArray(ComponentId_t id, Params& params, TimeConverter* tc, Event::HandlerBase* handler)
        : TileUnitArrayModel<T>(id, params, tc, handler) {}
};
}}
#endif
