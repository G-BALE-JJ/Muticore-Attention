#ifndef SST_GOLEM_DECODER_MEMORY_LAYOUT_H
#define SST_GOLEM_DECODER_MEMORY_LAYOUT_H
#include <cstdint>
#include <vector>
namespace SST { namespace Golem {
// One address allocator shared by the guest, accelerator, and host manifest
// dumper. Layers own weights; activations alternate between two slots; all
// intermediate tensors and completion flags are reused after the ACK barrier.
struct DecoderLayerMemory {
    uint64_t input, output, gamma1, gamma2, qkv_weight;
    uint64_t wo_weight, gate_weight, up_weight, down_weight;
};
struct DecoderMemoryRegion {
    const char* name;
    uint32_t layer;
    uint64_t offset, bytes;
};
struct DecoderMemoryLayout {
    static constexpr uint64_t node_stride = 0x08000000ull;
    static constexpr uint64_t gm_stride = 0x00200000ull;
    static constexpr uint64_t projection_release = 0x1008;
    static constexpr uint64_t decoder_release = 0x1010;
    static constexpr uint64_t layer_release = 0x1018;
    static constexpr uint64_t final_norm_release = 0x1020;
    static constexpr uint64_t lm_head_release = 0x1028;
    static constexpr uint64_t embedding_release = 0x1030;
    static constexpr uint64_t descriptor = 0x3000;
    static constexpr uint64_t topology = 0x1100;
    static constexpr uint64_t attention_descriptor = 0x2000;
    static constexpr uint64_t projection_scratch = 0x8000;
    static constexpr uint64_t decoder_input_gm=0x10000;
    static constexpr uint64_t decoder_weight_gm=0x110000;
    static constexpr uint64_t decoder_output_gm=0x112000;
    static constexpr uint64_t decoder_required_gm=0x118000+64;
    uint64_t activation[2]{}, projection_done=0, decoder_done=0;
    uint64_t norm1=0,q=0,k=0,v=0,o=0,q_panel=0,k_panel=0,v_panel=0,rope=0;
    uint64_t wo=0,h1=0,norm2=0,gate=0,up=0,act=0,gated=0,down=0,used_bytes=0;
    uint64_t final_gamma=0, final_output=0, final_done=0;
    uint64_t hbm_node_stride=node_stride, lm_weight=0, lm_output=0, lm_done=0;
    uint32_t vocab=0;
    uint64_t embedding_tokens=0,embedding_weight=0,embedding_output=0,embedding_done=0;
    bool valid=false;
    std::vector<DecoderLayerMemory> layers;
    std::vector<DecoderMemoryRegion> regions;
};
inline DecoderMemoryLayout makeDecoderMemoryLayout(uint32_t sequence,
        uint32_t hq,uint32_t hkv,uint32_t dim,uint32_t count,bool final_norm=false,
        uint32_t vocab=0,uint64_t node_bytes=DecoderMemoryLayout::node_stride,bool embedding=false) {
    DecoderMemoryLayout m;
    m.hbm_node_stride=node_bytes; m.vocab=vocab;
    const uint64_t hidden=uint64_t(hq)*dim, intermediate=4*hidden;
    if (sequence!=1024 || dim!=64 || (hidden!=256 && hidden!=2048) ||
        !hkv || hq%hkv || !count || count>32 || (vocab && !final_norm) || (embedding && !vocab) || vocab>1048576 ||
        node_bytes<m.node_stride || (node_bytes & (node_bytes-1))) return m;
    const uint64_t qbytes=uint64_t(sequence/4)*hidden*2;
    const uint64_t kvbytes=uint64_t(sequence/4)*hkv*dim*2;
    auto alloc=[&](const char* name,uint64_t bytes,uint32_t layer=0) {
        const uint64_t offset=(m.used_bytes+4095)&~uint64_t(4095);
        m.used_bytes=offset+bytes;
        m.regions.push_back({name,layer,offset,bytes});
        return offset;
    };
    m.activation[0]=alloc("activation0",qbytes);
    m.activation[1]=alloc("activation1",qbytes);
    m.projection_done=alloc("projection_done",32);
    m.decoder_done=alloc("decoder_done",32);
    m.norm1=alloc("norm1",qbytes); m.q=alloc("q",qbytes);
    m.k=alloc("k",kvbytes); m.v=alloc("v",kvbytes); m.o=alloc("o",qbytes);
    m.q_panel=alloc("q_panel",qbytes); m.k_panel=alloc("k_panel",kvbytes);
    m.v_panel=alloc("v_panel",kvbytes);
    m.rope=alloc("rope",uint64_t(sequence)*dim*2);
    m.wo=alloc("wo",qbytes); m.h1=alloc("h1",qbytes); m.norm2=alloc("norm2",qbytes);
    m.gate=alloc("gate",4*qbytes); m.up=alloc("up",4*qbytes);
    m.act=alloc("act",4*qbytes); m.gated=alloc("gated",4*qbytes);
    m.down=alloc("down",qbytes);
    for (uint32_t layer=0;layer<count;++layer) {
        DecoderLayerMemory l{};
        l.input=m.activation[layer%2]; l.output=m.activation[(layer+1)%2];
        l.gamma1=alloc("gamma1",hidden*2,layer+1);
        l.gamma2=alloc("gamma2",hidden*2,layer+1);
        // QKV weights are replicated; other linear weights stripe output tiles.
        l.qkv_weight=alloc("qkv_weight",(hq+2*hkv)*dim*hidden*2,layer+1);
        l.wo_weight=alloc("wo_weight",hidden*hidden*2/4,layer+1);
        l.gate_weight=alloc("gate_weight",intermediate*hidden*2/4,layer+1);
        l.up_weight=alloc("up_weight",intermediate*hidden*2/4,layer+1);
        l.down_weight=alloc("down_weight",hidden*intermediate*2/4,layer+1);
        m.layers.push_back(l);
    }
    if (final_norm) {
        m.final_gamma=alloc("final_gamma",hidden*2);
        m.final_output=alloc("final_output",qbytes);
        m.final_done=alloc("final_done",32);
    }
    if (vocab) {
        // Pad the vocabulary to 64-column tiles, then reserve the largest
        // stripe on every node. Only valid logits are returned by the host.
        const uint64_t tiles=(uint64_t(vocab)+63)/64;
        m.lm_weight=alloc("lm_weight",((tiles+3)/4)*hidden*64*2);
        m.lm_output=alloc("lm_output",((uint64_t(vocab)+63)/64)*64*2);
        m.lm_done=alloc("lm_done",32);
    }
    if (embedding) {
        m.embedding_tokens=alloc("embedding_tokens",uint64_t(sequence/4)*4);
        m.embedding_weight=alloc("embedding_weight",((uint64_t(vocab)+3)/4)*hidden*2);
        m.embedding_output=alloc("embedding_output",qbytes);
        m.embedding_done=alloc("embedding_done",32);
        m.layers.front().input=m.embedding_output;
    }
    m.valid=m.used_bytes<=m.hbm_node_stride;
    return m;
}
}} // namespace SST::Golem
#endif
