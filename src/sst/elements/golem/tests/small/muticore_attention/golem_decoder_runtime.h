#ifndef GOLEM_DECODER_RUNTIME_H
#define GOLEM_DECODER_RUNTIME_H
#include "../../../attention/decoderOpAbi.h"
#include "../../../attention/decoderMemoryLayout.h"
#ifndef GOLEM_DECODER_STAGE
#define GOLEM_DECODER_STAGE 0
#endif
static bool submit_decoder_op(uint32_t worker,const SST::Golem::DecoderOpDesc& d) {
    const uint64_t desc=uint64_t(worker)*SST::Golem::DecoderMemoryLayout::gm_stride+SST::Golem::DecoderMemoryLayout::descriptor;
    const auto* words=reinterpret_cast<const uint64_t*>(&d);
    for (uint32_t i=0;i<sizeof(d)/8;++i) reg2gm(words[i],desc+i*8);
    uint64_t status;
    asm volatile(".insn r 0x0b, 7, 0x26, %0, %1, x0" : "=r"(status) : "r"(desc) : "memory");
    if (status) return false;
    asm volatile(".insn r 0x0b, 7, 0x27, %0, x0, x0" : "=r"(status) : : "memory");
    return status==0;
}
static bool run_final_norm(uint32_t worker,uint32_t rows,uint32_t hidden,const SST::Golem::DecoderMemoryLayout& layout) {
    using namespace SST::Golem;
    if (!layout.valid || !layout.final_done || layout.layers.empty()) return false;
    DecoderOpDesc d{};
    d.magic=GOLEM_DECODER_OP_MAGIC; d.size_bytes=sizeof(d);
    d.op=static_cast<uint32_t>(DecoderOp::RmsNorm); d.stage=10;
    d.epoch=layout.layers.size()+1; d.rows=rows; d.input_dim=hidden; d.output_dim=hidden;
    d.manager=(worker-4)%4; d.lane=(worker-4)/4;
    const uint64_t node=uint64_t(d.manager+1)*layout.hbm_node_stride;
    d.input_addr=node+layout.layers.back().output; d.input1_addr=node+layout.final_gamma;
    d.output_addr=node+layout.final_output; d.node_stride=layout.hbm_node_stride;
    d.completion_addr=node+layout.final_done+d.lane*8;
    return submit_decoder_op(worker,d);
}
static bool run_embedding(uint32_t worker,uint32_t rows,uint32_t hidden,const SST::Golem::DecoderMemoryLayout& layout) {
    using namespace SST::Golem;
    DecoderOpDesc d{};
    d.magic=GOLEM_DECODER_OP_MAGIC; d.size_bytes=sizeof(d);
    d.op=static_cast<uint32_t>(DecoderOp::Embedding); d.stage=12; d.epoch=1;
    d.rows=rows; d.input_dim=layout.vocab; d.output_dim=hidden;
    d.manager=(worker-4)%4; d.lane=(worker-4)/4;
    const uint64_t node=uint64_t(d.manager+1)*layout.hbm_node_stride;
    d.input_addr=node+layout.embedding_tokens; d.output_addr=node+layout.embedding_output;
    d.weights_offset=layout.embedding_weight; d.node_stride=layout.hbm_node_stride;
    d.completion_addr=node+layout.embedding_done+d.lane*8;
    return submit_decoder_op(worker,d);
}
static bool run_lm_head(uint32_t worker,uint32_t hidden,const SST::Golem::DecoderMemoryLayout& layout) {
    using namespace SST::Golem;
    const uint32_t columns=((layout.vocab+63)/64)*64;
    for (uint32_t first=0;first<columns;first+=8192) {
        DecoderOpDesc d{};
        d.magic=GOLEM_DECODER_OP_MAGIC; d.size_bytes=sizeof(d);
        d.op=static_cast<uint32_t>(DecoderOp::Linear); d.stage=11;
        d.epoch=layout.layers.size()+2+first/8192; d.rows=1;
        d.input_dim=hidden; d.output_dim=std::min(8192u,columns-first);
        d.manager=(worker-4)%4; d.lane=(worker-4)/4; d.output_tile_begin=first/64;
        // Token S-1 belongs to the fourth data node; all workers read it.
        d.input_addr=4*layout.hbm_node_stride+layout.final_output+255ull*hidden*2;
        d.output_addr=(d.manager+1)*layout.hbm_node_stride+layout.lm_output+first*2;
        d.weights_offset=layout.lm_weight; d.node_stride=layout.hbm_node_stride;
        if (first+d.output_dim==columns)
            d.completion_addr=(d.manager+1)*layout.hbm_node_stride+layout.lm_done+d.lane*8;
        if (!submit_decoder_op(worker,d)) return false;
    }
    return true;
}
#if GOLEM_DECODER_STAGE < 0 || GOLEM_DECODER_STAGE > 4
#error "decoder stage must be 0 (legacy), 1 (Wo), 2 (residual), 3 (FFN), or 4 (layer)"
#endif
static bool run_decoder(uint32_t worker,uint32_t rows,uint32_t hidden,uint32_t dim,uint64_t o_offset,
        const SST::Golem::DecoderMemoryLayout* layout=nullptr,uint32_t layer=0) {
    using namespace SST::Golem;
    // This first integration preserves all Attention audit buffers at S=1024.
    // Larger sequences require a new HBM allocator rather than silent overlap.
    if (rows != 256 || hidden > 2048 || hidden % 256 || dim != 64) return false;
    const uint32_t manager=(worker-4)%4, lane=(worker-4)/4;
    const uint64_t stride=layout?layout->hbm_node_stride:SST::Golem::DecoderMemoryLayout::node_stride;
    const uint64_t node=uint64_t(manager+1)*stride;
    DecoderLayerMemory l{};
    if (layout) l=layout->layers.at(layer);
    else { l.input=0; l.output=DEC_Y; l.gamma2=DEC_GAMMA;
        l.wo_weight=DEC_WO_WEIGHT; l.gate_weight=DEC_GATE_WEIGHT;
        l.up_weight=DEC_UP_WEIGHT; l.down_weight=DEC_DOWN_WEIGHT; }
    const uint64_t wo=layout?layout->wo:DEC_WO, h1=layout?layout->h1:DEC_H1;
    const uint64_t norm=layout?layout->norm2:DEC_NORM, gate=layout?layout->gate:DEC_GATE;
    const uint64_t up=layout?layout->up:DEC_UP, act=layout?layout->act:DEC_ACT;
    const uint64_t gated=layout?layout->gated:DEC_GATED, down=layout?layout->down:DEC_DOWN;
    const auto issue=[&](DecoderOp op,uint32_t stage,uint32_t k,uint32_t n,uint64_t input,
        uint64_t input1,uint64_t output,uint64_t weights,uint32_t head,bool last) {
        DecoderOpDesc d{};
        d.magic=GOLEM_DECODER_OP_MAGIC; d.size_bytes=sizeof(d); d.op=static_cast<uint32_t>(op); d.stage=stage;
        d.epoch=layout?layer+1:1; d.rows=rows; d.input_dim=k; d.output_dim=n; d.manager=manager; d.lane=lane; d.head_dim=head;
        d.input_addr=node+input; d.input1_addr=(op==DecoderOp::Add || op==DecoderOp::Multiply || op==DecoderOp::RmsNorm)?node+input1:0; d.output_addr=node+output;
        d.weights_offset=weights; d.node_stride=stride;
        d.completion_addr=last?node+(layout?layout->decoder_done:DEC_DONE)+lane*8:0;
        return submit_decoder_op(worker,d);
    };
    if (!issue(DecoderOp::Linear,1,hidden,hidden,o_offset,0,wo,l.wo_weight,0,GOLEM_DECODER_STAGE==1)) return false;
    if (GOLEM_DECODER_STAGE==1) return true;
    // X is the original input, before the first RMSNorm.
    if (!issue(DecoderOp::Add,2,hidden,hidden,wo,l.input,h1,0,0,GOLEM_DECODER_STAGE==2)) return false;
    if (GOLEM_DECODER_STAGE==2) return true;
    const uint32_t intermediate=4*hidden;
    if (!issue(DecoderOp::RmsNorm,3,hidden,hidden,h1,l.gamma2,norm,0,0,false) ||
        !issue(DecoderOp::Linear,4,hidden,intermediate,norm,0,gate,l.gate_weight,0,false) ||
        !issue(DecoderOp::Linear,5,hidden,intermediate,norm,0,up,l.up_weight,0,false) ||
        !issue(DecoderOp::Silu,6,intermediate,intermediate,gate,0,act,0,0,false) ||
        !issue(DecoderOp::Multiply,7,intermediate,intermediate,act,up,gated,0,0,false) ||
        !issue(DecoderOp::Linear,8,intermediate,hidden,gated,0,down,l.down_weight,0,GOLEM_DECODER_STAGE==3)) return false;
    if (GOLEM_DECODER_STAGE==3) return true;
    return issue(DecoderOp::Add,9,hidden,hidden,down,h1,l.output,0,0,true);
}
#endif
