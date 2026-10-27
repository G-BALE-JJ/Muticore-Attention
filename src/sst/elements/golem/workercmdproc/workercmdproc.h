#ifndef _H_GOLEM_WORKER_COMMAND_PROCESSOR
#define _H_GOLEM_WORKER_COMMAND_PROCESSOR
#include <sst/elements/golem/tileengine/tileEngine.h>
namespace SST { namespace Golem {
// Source and ELI compatibility: existing clients execute the same TE and
// retain all timed scheduling, ownership and LLM completion callbacks.
class WorkerCommandProcessorLocal : public TileEngineLocal {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(WorkerCommandProcessorLocal,"golem",
        "WorkerCommandProcessorLocal",SST_ELI_ELEMENT_VERSION(1,0,0),
        "Compatibility entry for the Tile Engine",SST::Golem::WorkerCommandProcessorAPI)
    WorkerCommandProcessorLocal(ComponentId_t id,Params& params):TileEngineLocal(id,params) {}
};
}}
#endif
