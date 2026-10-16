#include "../../../attention/decoderMemoryLayout.h"
#include <cstdio>
#include <cstdlib>
int main(int argc,char** argv) {
    if (argc<6 || argc>11) return 1;
    auto m=SST::Golem::makeDecoderMemoryLayout(std::atoi(argv[1]),std::atoi(argv[2]),
        std::atoi(argv[3]),std::atoi(argv[4]),std::atoi(argv[5]),argc>=7 && std::atoi(argv[6])==1,
        argc>=8 ? std::strtoul(argv[7],nullptr,10):0,
        argc>=9 ? std::strtoull(argv[8],nullptr,10):SST::Golem::DecoderMemoryLayout::node_stride,
        argc>=10 && std::atoi(argv[9])==1,argc>=11 && std::atoi(argv[10])==1);
    if (!m.valid) { std::fprintf(stderr,"decoder tensors/weights exceed supported HBM layout (used=%llu)\n",(unsigned long long)m.used_bytes); return 2; }
    std::printf("{\"node_stride\":%llu,\"gm_stride\":%llu,\"used_bytes\":%llu,\"shared\":{",(unsigned long long)m.hbm_node_stride,(unsigned long long)m.gm_stride,(unsigned long long)m.used_bytes);
#define FIELD(n) std::printf("\"" #n "\":%llu,",(unsigned long long)m.n)
    FIELD(projection_done); FIELD(decoder_done); FIELD(norm1); FIELD(q); FIELD(k); FIELD(v); FIELD(o);
    FIELD(q_panel); FIELD(k_panel); FIELD(v_panel); FIELD(rope);
    FIELD(wo); FIELD(h1); FIELD(norm2); FIELD(gate); FIELD(up); FIELD(act); FIELD(gated);
    std::printf("\"down\":%llu},\"activations\":[%llu,%llu],\"layers\":[",(unsigned long long)m.down,(unsigned long long)m.activation[0],(unsigned long long)m.activation[1]);
#undef FIELD
    for (size_t i=0;i<m.layers.size();++i) {
        auto& l=m.layers[i]; if (i) std::printf(","); std::printf("{");
#define FIELD(n) std::printf("\"" #n "\":%llu,",(unsigned long long)l.n)
        FIELD(input); FIELD(output); FIELD(gamma1); FIELD(gamma2); FIELD(qkv_weight);
        FIELD(wo_weight); FIELD(gate_weight); FIELD(up_weight);
        std::printf("\"down_weight\":%llu}",(unsigned long long)l.down_weight);
#undef FIELD
    }
    std::printf("],\"regions\":[");
    for (size_t i=0;i<m.regions.size();++i) {
        auto& r=m.regions[i];
        std::printf("%s{\"name\":\"%s\",\"layer\":%u,\"offset\":%llu,\"bytes\":%llu}",i?",":"",r.name,r.layer,(unsigned long long)r.offset,(unsigned long long)r.bytes);
    }
    std::printf("]");
    if (m.final_done) std::printf(",\"final_norm\":{\"gamma\":%llu,\"output\":%llu,\"done\":%llu}",
        (unsigned long long)m.final_gamma,(unsigned long long)m.final_output,(unsigned long long)m.final_done);
    if (m.vocab) std::printf(",\"lm_head\":{\"vocab\":%u,\"weight\":%llu,\"output\":%llu,\"done\":%llu}",
        m.vocab,(unsigned long long)m.lm_weight,(unsigned long long)m.lm_output,(unsigned long long)m.lm_done);
    if (m.embedding_done) std::printf(",\"embedding\":{\"tokens\":%llu,\"weight\":%llu,\"output\":%llu,\"done\":%llu}",
        (unsigned long long)m.embedding_tokens,(unsigned long long)m.embedding_weight,(unsigned long long)m.embedding_output,(unsigned long long)m.embedding_done);
    std::printf("}\n");
}
