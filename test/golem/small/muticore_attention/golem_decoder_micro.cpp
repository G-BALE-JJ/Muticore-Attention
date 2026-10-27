// Direct production DecoderOpJob entry: no Attention/projection/startup polling.
#include "golem_attention_runtime.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include "golem_decoder_runtime.h"

int main() {
    using namespace SST::Golem;
    const int core=sched_getcpu();
    const char* config=std::getenv("GOLEM_DECODER_MICRO_CONFIG");
    unsigned workers=0,rows=0,hidden=0,fused=0,down=0,residual=0;
    char extra;
    if (!config || std::sscanf(config,"%u,%u,%u,%u,%u,%u%c",&workers,&rows,&hidden,&fused,&down,&residual,&extra)!=6 ||
        (workers!=1 && workers!=4 && workers!=16) ||
        rows!=(workers==16?256u:64u) || (hidden!=256 && hidden!=2048) || fused>3 || down>1 || residual>1 || (fused>=2 && !down) || (fused==3 && !residual)) return 1;
    if (core<4 || core>=int(4+workers)) return 0;
    const auto layout=makeDecoderMemoryLayout(1024,hidden/64,hidden==256?2:8,64,2);
    if (!layout.valid) return 1;
    const auto& l=layout.layers.front();
    const uint32_t manager=(core-4)%4,lane=(core-4)/4;
    const uint64_t node=uint64_t(manager+1)*layout.hbm_node_stride;
    const uint32_t intermediate=4*hidden;
    const auto issue=[&](DecoderOp op,uint32_t stage,uint32_t k,uint32_t n,
        uint64_t input,uint64_t input1,uint64_t output,uint64_t weights) {
        DecoderOpDesc d{};
        d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);
        d.op=static_cast<uint32_t>(op);d.stage=stage;d.epoch=1;
        d.rows=rows;d.input_dim=k;d.output_dim=n;d.manager=manager;d.lane=lane;
        d.input_addr=node+input;d.output_addr=node+output;
        d.input1_addr=op==DecoderOp::GateUpSiluMultiply?input1:input1?node+input1:0;
        d.weights_offset=weights;d.node_stride=layout.hbm_node_stride;
        return submit_decoder_op(core,d);
    };
    bool ok;
    if (fused>=2) {
        DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);
        d.op=static_cast<uint32_t>(DecoderOp::GateUpSiluDown);d.stage=4;d.epoch=1;
        d.rows=rows;d.input_dim=hidden;d.output_dim=intermediate;d.manager=manager;d.lane=lane;
        d.input_addr=node+layout.norm2;d.input1_addr=l.up_weight;d.weights_offset=l.gate_weight;
        d.down_weights_offset=static_cast<uint32_t>(l.down_weight);d.output_addr=node+layout.down;
        if (fused==3) d.residual_offset=static_cast<uint32_t>(layout.h1);
        d.node_stride=layout.hbm_node_stride;ok=submit_decoder_op(core,d);
    } else if (fused) ok=issue(DecoderOp::GateUpSiluMultiply,4,hidden,intermediate,
        layout.norm2,l.up_weight,layout.gated,l.gate_weight);
    else ok=issue(DecoderOp::Linear,4,hidden,intermediate,layout.norm2,0,layout.gate,l.gate_weight) &&
        issue(DecoderOp::Linear,5,hidden,intermediate,layout.norm2,0,layout.up,l.up_weight) &&
        issue(DecoderOp::SiluMultiply,7,intermediate,intermediate,layout.gate,layout.up,layout.gated,0);
    if (ok && down && fused<2) ok=issue(DecoderOp::Linear,8,intermediate,hidden,layout.gated,0,layout.down,l.down_weight);
    if (ok && residual && fused!=3) ok=issue(DecoderOp::Add,9,hidden,hidden,layout.down,layout.h1,layout.down,0);
    std::printf("DECODER_MICRO core=%d status=%u workers=%u rows=%u hidden=%u fused=%u down=%u\n",
        core,unsigned(!ok),workers,rows,hidden,fused,down);
    return ok?0:1;
}
