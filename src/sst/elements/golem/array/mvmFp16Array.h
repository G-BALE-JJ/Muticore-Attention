#ifndef SST_GOLEM_MVM_FP16_ARRAY_H
#define SST_GOLEM_MVM_FP16_ARRAY_H

#include <sst/core/component.h>
#include <sst/elements/golem/array/mvmComputeArray.h>
#include <sst/elements/golem/fp16.h>

namespace SST { namespace Golem {

class MVMFp16Array : public MVMComputeArray<GolemFp16> {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(
        MVMFp16Array,
        "golem",
        "MVMFp16Array",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Compute array with FP16 operands and FP16 accumulation",
        SST::Golem::MVMComputeArray<SST::Golem::GolemFp16>
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"arrayLatency", "Latency of array computation", "100ns"},
        {"verbose", "Set component verbosity", "0"},
        {"max_instructions", "Maximum queued instructions", "8"},
        {"clock", "Component clock", "1GHz"},
        {"mmioAddr", "MMIO address"},
        {"numArrays", "Number of arrays", "1"},
        {"arrayInputSize", "Input vector length"},
        {"arrayOutputSize", "Output vector length"},
        {"inputOperandSize", "Input operand bytes", "2"},
        {"outputOperandSize", "Output operand bytes", "2"},
        {"functionalCompute", "Execute numerical MACs", "1"},
        {"attention_cluster_qk_arrays", "First PV array id", "16"},
    )

    MVMFp16Array(ComponentId_t id, Params& params,
                 TimeConverter* tc, Event::HandlerBase* handler)
        : MVMComputeArray<GolemFp16>(id, params, tc, handler) {}
};

}} // namespace SST::Golem
#endif
