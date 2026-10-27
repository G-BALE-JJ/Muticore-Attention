#ifndef GOLEM_TAIL_SCHEDULE_H
#define GOLEM_TAIL_SCHEDULE_H
#include <cstdint>

// Software ordering only. All tensor jobs and notifications use existing
// descriptors, local GM polling and timed remote_store instructions.
static uint32_t final_tail_producer(uint32_t logical, uint32_t sequence) {
    if (!logical || logical>sequence || !sequence || sequence>1024 || sequence%256) return 0;
    const uint32_t rows=sequence/4,token=logical-1;
    return 4+token/rows+4*((token%rows)/64);
}

template<class Local, class Remote>
static void publish_final_tail(uint32_t core, uint64_t stride, uint64_t offset,
                               uint64_t epoch, Local local, Remote remote) {
    const uint64_t source=uint64_t(core)*stride+offset;
    local(epoch,source);
    for(uint32_t target=4;target<20;++target)if(target!=core)
        remote(source,uint64_t(target)*stride+offset);
}

template<class Norm, class Publish, class Wait, class Lm>
static bool run_final_tail(uint32_t core, uint32_t producer, bool stream, bool producer_first,
                           Norm norm, Publish publish, Wait wait, Lm lm) {
    // A separate, coarse scheduling experiment: prioritize the whole existing
    // 64-row producer job. No row is omitted, duplicated or recomputed.
    if (stream && producer_first && core!=producer) wait();
    if (!norm()) return false;
    if (stream && core==producer) publish();
    wait();
    return lm();
}
#endif
