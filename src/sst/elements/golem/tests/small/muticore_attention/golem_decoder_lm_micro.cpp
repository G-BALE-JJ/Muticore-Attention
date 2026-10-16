// Isolated last-token LM head: same descriptors, ownership and chunks as full prefill.
#include "golem_attention_runtime.h"
#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "golem_decoder_runtime.h"
int main() {
    using namespace SST::Golem;
    unsigned workers=0,hidden=0,vocab=0,nodeMiB=0;char extra;
    const char* config=std::getenv("GOLEM_DECODER_MICRO_CONFIG");
    if (!config || std::sscanf(config,"%u,%u,%u,%u%c",&workers,&hidden,&vocab,&nodeMiB,&extra)!=4 ||
        (workers!=1 && workers!=16) || (hidden!=256 && hidden!=2048) ||
        !vocab || vocab>128256 || nodeMiB!=256) return 1;
    const int core=sched_getcpu();
    if (core<4 || core>=int(4+workers)) return 0;
    const auto layout=makeDecoderMemoryLayout(1024,hidden/64,hidden==256?2:8,64,2,true,vocab,uint64_t(nodeMiB)<<20);
    if (!layout.valid) return 1;
    const bool ok=run_lm_head(core,hidden,layout);
    std::printf("LM_MICRO core=%d status=%u workers=%u hidden=%u vocab=%u\n",core,unsigned(!ok),workers,hidden,vocab);
    return ok?0:1;
}
