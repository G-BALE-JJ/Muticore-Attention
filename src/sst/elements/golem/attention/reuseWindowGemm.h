#ifndef SST_GOLEM_REUSE_WINDOW_GEMM_H
#define SST_GOLEM_REUSE_WINDOW_GEMM_H
#include <sst/elements/golem/globalmemory/globalmemory.h>
#include <sst/elements/golem/workercmdproc/workercmdproc.h>
#include <sst/elements/golem/fp16.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>
namespace SST { namespace Golem {
// Four N tiles share an explicitly allocated GM input window. Each output
// remains in the arrays for an entire K chunk. Only chunk boundaries spill C,
// through timed GM ports. DMA/weight reads run independently of array compute.
// Transfer snapshots are bounded by their GM slots, not an extra hardware cache.
class ReuseWindowGemm {
public:
    using Values=std::vector<double>;
    using Loaded=std::function<void(bool,const Values&)>;
    using Input=std::function<void(uint32_t,Loaded)>;
    using Initial=std::function<void(uint32_t,Loaded)>;
    using Output=std::function<void(uint32_t,const Values&)>;
    using Acquire=std::function<bool()>;
    using Release=std::function<void()>;
    using Fetch=std::function<void(uint64_t,uint64_t,std::function<void(bool)>)>;
    ReuseWindowGemm(GlobalMemoryAPI* gm,WorkerCommandProcessorAPI* wcp):gm_(gm),wcp_(wcp) {}
    bool active() const {return active_;}
    uint32_t kWindow() const {return kWindow_;}
    uint64_t localReadBytes() const{return localRead_;}
    uint64_t localWriteBytes() const{return localWrite_;}
    uint64_t hbmReadBytes() const{return hbmRead_;}
    uint64_t windows() const{return windows_;}
    uint64_t launches() const{return launches_;}
    uint64_t prefetchRequests() const{return prefetchRequests_;}
    uint64_t prefetchHits() const{return prefetchHits_;}
    uint64_t cReadBytes() const{return cRead_;}
    uint64_t cWriteBytes() const{return cWrite_;}
    uint64_t drains() const{return drains_;}
    uint64_t restores() const{return restores_;}
    uint64_t groupLaunches() const{return groupLaunches_;}
    uint64_t groupOverlapCycles() const{return groupOverlap_;}
    uint64_t groupProgramCycles() const{return groupProgram_;}
    uint64_t groupComputeArrayCycles() const{return groupCompute_;}
    uint64_t transferOverlapCycles() const{return transferOverlap_;}
    bool computing() const {return remaining_[0] || remaining_[1];}
    bool transferring() const {for(const auto& s:slots_)if(s.valid && (s.dma || s.reading || s.pending))return true;
        for(const auto& s:inputSlots_)if(s.valid && !s.ready)return true;return inputReading_ || inputPending_;}
    uint64_t leaseWaitCycles() const {return leaseWait_;}
    void trace(uint32_t core,uint32_t epoch,uint32_t stage,const char* name) const {
        uint32_t inputLive=0;for(const auto& s:inputSlots_)if(s.valid)++inputLive;
        uint32_t weightLive=0;for(const auto& s:slots_)if(s.valid)++weightLive;
        std::printf("[GEMM_WINDOW_BRIDGE] core=%u epoch=%u stage=%u operator=%s enabled=%u slots=%u seam_prefetches=%llu live_at_finish=%u capacity_added=0\n",
            core,epoch,stage,name,bridgeWeights_?1u:0u,slotCount_,(unsigned long long)bridgePrefetches_,weightLive);
        std::printf("[GEMM_A_PIPELINE] core=%u epoch=%u stage=%u operator=%s enabled=%u entries=2 high_water=%llu ready_hits=%llu live_at_finish=%u\n",
            core,epoch,stage,name,inputPrefetch_?1u:0u,(unsigned long long)inputHighWater_,
            (unsigned long long)inputHits_,inputLive);
        std::printf("[GEMM_WINDOW_FLOW] core=%u epoch=%u stage=%u operator=%s groups=%u early_a=%u output_overlap=%u gm_lookahead=%u a_first=%u b_gm_deferred=%llu active_cycles=%llu wait_b_dma=%llu wait_b_gm=%llu wait_a=%llu wait_array=%llu wait_output=%llu output_pending=%llu output_compute_overlap=%llu phase_load=%llu phase_restore=%llu phase_acquire=%llu phase_matrix=%llu phase_scatter=%llu phase_launch=%llu phase_drain=%llu phase_output=%llu phase_finish=%llu\n",
            core,epoch,stage,name,groups_,earlyA_?1:0,detachOutput_?1:0,gmLookahead_,aFirst_?1:0,
            (unsigned long long)gmDeferred_,
            (unsigned long long)activeCycles_,(unsigned long long)waitDma_,(unsigned long long)waitGm_,
            (unsigned long long)waitA_,(unsigned long long)waitArray_,(unsigned long long)waitOutput_,
            (unsigned long long)outputPendingCycles_,(unsigned long long)outputComputeOverlap_,
            (unsigned long long)phases_[0],(unsigned long long)phases_[1],(unsigned long long)phases_[2],
            (unsigned long long)phases_[3],(unsigned long long)phases_[4],(unsigned long long)phases_[5],
            (unsigned long long)phases_[6],(unsigned long long)phases_[7],(unsigned long long)phases_[8]);
    }
    void start(uint32_t rows,uint32_t kTiles,std::vector<uint32_t> tiles,
               uint64_t aScratch,uint64_t bScratch,uint64_t cScratch,
               Input input,std::function<uint64_t(uint32_t,uint32_t)> weight,
               Output output,std::function<void(bool)> done,Initial initial={},
               std::function<uint64_t(uint32_t)> inputAddress={},bool independentOutput=false,Fetch fetch={},
               Acquire acquire={},Release release={}) {
        rows_=rows;kTiles_=kTiles;tiles_=std::move(tiles);aBase_=aScratch;bBase_=bScratch;cBase_=cScratch;
        input_=std::move(input);weight_=std::move(weight);output_=std::move(output);
        done_=std::move(done);initial_=std::move(initial);
        inputAddress_=std::move(inputAddress);
        fetch_=std::move(fetch);acquire_=std::move(acquire);release_=std::move(release);leaseHeld_=false;leaseWait_=0;
        kWindow_=option("GOLEM_REUSE_WINDOW_K_TILES",32,1,32);
        bridgeWeights_=kTiles_<=kWindow_ && option("GOLEM_REUSE_WINDOW_B_WINDOW_BRIDGE",0,0,1);
        slotCount_=option("GOLEM_REUSE_WINDOW_PREFETCH_SLOTS",4,1,4);
        gmLookahead_=std::min(slotCount_,option("GOLEM_REUSE_WINDOW_GM_LOOKAHEAD",4,1,4));
        aFirst_=option("GOLEM_REUSE_WINDOW_A_FIRST",0,0,1);
        inputPrefetch_=option("GOLEM_REUSE_WINDOW_A_PREFETCH",0,0,1);
        inputSlots_={};inputHighWater_=inputHits_=0;
        gmDeferred_=0;
        groups_=rows_>32?option("GOLEM_REUSE_WINDOW_ARRAY_GROUPS",2,1,2):1;
        scatterWave_=rows_>32 && option("GOLEM_REUSE_WINDOW_SCATTER_WAVE",0,0,1);
        if(scatterWave_)groups_=2;
        earlyA_=option("GOLEM_REUSE_WINDOW_EARLY_A",0,0,1);
        lateLease_=option("GOLEM_REUSE_WINDOW_LATE_LEASE",0,0,1);
        // A detached consumer owns its output snapshot. Initial-C callbacks
        // may share the parent's writer state, so those remain synchronized.
        detachOutput_=independentOutput && !initial_ && option("GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP",0,0,1);
        arrays_.resize(rows);for(uint32_t i=0;i<rows;++i)arrays_[i]=i;
        const uint64_t aBytes=uint64_t(std::min(kWindow_,kTiles_))*rows_*128;
        const uint64_t bBytes=slotCount_*8192,cBytes=uint64_t(std::min<size_t>(4,tiles_.size()))*rows_*256;
        auto fits=[this](uint64_t addr,uint64_t bytes){return addr>=gm_->getBaseAddr() &&
            addr-gm_->getBaseAddr()<=gm_->getSize()-64 && bytes<=gm_->getSize()-64-(addr-gm_->getBaseAddr());};
        auto overlaps=[](uint64_t a,uint64_t x,uint64_t b,uint64_t y){return a<b+y && b<a+x;};
        if(!rows || rows>64 || !kTiles || tiles_.empty() || !fits(aBase_,aBytes) || !fits(bBase_,bBytes) ||
           !fits(cBase_,cBytes) || overlaps(aBase_,aBytes,bBase_,bBytes) || overlaps(aBase_,aBytes,cBase_,cBytes) ||
           overlaps(bBase_,bBytes,cBase_,cBytes)) {done_(false);return;}
        active_=true;pending_=false;outputWaiting_=false;nBase_=kBase_=n_=k_=0;cReadyCycle_=0;
        localRead_=localWrite_=hbmRead_=windows_=launches_=0;
        prefetchRequests_=prefetchHits_=cRead_=cWrite_=drains_=restores_=0;
        bridgePrefetches_=0;
        groupLaunches_=groupOverlap_=transferOverlap_=groupProgram_=groupCompute_=0;remaining_.fill(0);
        activeCycles_=waitDma_=waitGm_=waitA_=waitArray_=waitOutput_=0;
        outputPendingCycles_=outputComputeOverlap_=0;phases_.fill(0);phase_=0;outputAdvanced_=false;
        inputReading_=inputPending_=inputReady_=false;
        schedule([this]{beginWindow();});
    }
    void tick(uint64_t cycle) {
        cycle_=cycle;if(!active_)return;
        ++activeCycles_;++phases_[phase_];
        if(outputWaiting_){++outputPendingCycles_;if(remaining_[0] || remaining_[1])++outputComputeOverlap_;}
        if(remaining_[0] && remaining_[1])++groupOverlap_;
        bool transfer=false;
        if(aFirst_)pumpInput();
        for(uint32_t i=0;i<slotCount_;++i){auto& s=slots_[i];if(s.valid && (s.dma || s.reading || s.pending))transfer=true;
            // HBM prefetch retains all its existing slots. Only the local GM
            // snapshot reads are bounded by distance from the consuming tile,
            // so speculative B traffic cannot monopolize the single GM port.
            const uint32_t current=currentSequence();
            if(s.valid && s.reading && s.seq>=current+gmLookahead_){++gmDeferred_;continue;}
            pumpWeight(i);}
        if(!aFirst_)pumpInput();
        if((remaining_[0] || remaining_[1]) && transfer)++transferOverlap_;
        if(next_ && !pending_){auto f=std::move(next_);next_={};f();}
    }
    void resumeOutput(){
        if(!outputWaiting_)return;outputWaiting_=false;
        // Detached output released its snapshot before advancing. result_ can
        // now hold the next tile's restored C; its ACK must not clear that C.
        if(!outputAdvanced_){result_.clear();advanceN();}
    }
    void delayUntil(uint64_t cycle){cReadyCycle_=std::max(cReadyCycle_,cycle);}
private:
    struct WeightSlot {bool valid=false,dma=false,reading=false,pending=false,ready=false;uint32_t seq=0;size_t offset=0;std::vector<uint8_t> bytes;};
    struct InputSlot {bool valid=false,pending=false,ready=false;uint32_t seq=0;size_t offset=0;
        uint64_t address=0;std::vector<uint8_t> bytes;Values values;};
    static uint32_t option(const char* name,uint32_t fallback,uint32_t lo,uint32_t hi){
        const char* s=std::getenv(name);if(!s)return fallback;char* end=nullptr;auto v=std::strtoul(s,&end,10);
        if(!*s || *end || v<lo || v>hi)throw std::invalid_argument(name);return v;
    }
    void schedule(std::function<void()> f){next_=std::move(f);}
    void finish(bool ok){active_=false;next_={};pending_=false;auto f=std::move(done_);f(ok);}
    static std::vector<uint8_t> pack(const Values& values,uint32_t bytes){
        std::vector<uint8_t> raw(values.size()*bytes);for(size_t i=0;i<values.size();++i){
            if(bytes==2){uint16_t h=golem_float_to_fp16(static_cast<float>(values[i]));std::memcpy(raw.data()+i*2,&h,2);}
            else{float v=values[i];std::memcpy(raw.data()+i*4,&v,4);}}return raw;
    }
    static Values unpack(const std::vector<uint8_t>& raw,uint32_t bytes){
        Values v(raw.size()/bytes);for(size_t i=0;i<v.size();++i){if(bytes==2){uint16_t h;std::memcpy(&h,raw.data()+i*2,2);v[i]=golem_fp16_to_float(h);}
            else{float f;std::memcpy(&f,raw.data()+i*4,4);v[i]=f;}}return v;
    }
    void read(uint64_t addr,size_t bytes,std::function<void(const std::vector<uint8_t>&)> cb){
        ioAddr_=addr;io_.assign(bytes,0);offset_=0;readDone_=std::move(cb);schedule([this]{readChunk();});
    }
    void readChunk(){
        size_t len=std::min(gm_->localMaxRequestBytes(),io_.size()-offset_);pending_=true;
        if(!gm_->localReadAsync(ioAddr_+offset_,len,LocalMemoryClient::RoCC,++tag_,
            [this,len](bool ok,uint64_t,const std::vector<uint8_t>& b){pending_=false;
                if(!ok || b.size()!=len){finish(false);return;}std::copy(b.begin(),b.end(),io_.begin()+offset_);
                offset_+=len;localRead_+=len;if(offset_<io_.size())schedule([this]{readChunk();});
                else{auto cb=std::move(readDone_);auto b=std::move(io_);cb(b);}})){
            pending_=false;schedule([this]{readChunk();});}
    }
    void write(uint64_t addr,std::vector<uint8_t> bytes,std::function<void()> cb){
        ioAddr_=addr;io_=std::move(bytes);offset_=0;writeDone_=std::move(cb);schedule([this]{writeChunk();});
    }
    void writeChunk(){
        size_t len=std::min(gm_->localMaxRequestBytes(),io_.size()-offset_);
        std::vector<uint8_t> b(io_.begin()+offset_,io_.begin()+offset_+len);pending_=true;
        if(!gm_->localWriteAsync(ioAddr_+offset_,b,LocalMemoryClient::RoCC,++tag_,[this,len](bool ok,uint64_t){
            pending_=false;if(!ok){finish(false);return;}offset_+=len;localWrite_+=len;
            if(offset_<io_.size())schedule([this]{writeChunk();});else{io_.clear();auto cb=std::move(writeDone_);cb();}})){
            pending_=false;schedule([this]{writeChunk();});}
    }
    void beginWindow(){
        if(outputWaiting_){phase_=7;++waitOutput_;schedule([this]{beginWindow();});return;}
        phase_=0;
        ++windows_;nCount_=std::min<uint32_t>(4,tiles_.size()-nBase_);kCount_=std::min(kWindow_,kTiles_-kBase_);
        n_=k_=loadedA_=0;preparedSeq_=~uint32_t(0);
        if(!bridgeWeights_ || !nBase_){nextPrefetch_=0;for(auto& slot:slots_)slot=WeightSlot{};}
        fillPrefetch();schedule(inputAddress_?std::function<void()>([this]{restoreC();}):std::function<void()>([this]{loadA();}));
    }
    void fillPrefetch(){
        while(nextPrefetch_<sequenceLimit()){uint32_t index=nextPrefetch_%slotCount_;auto& s=slots_[index];if(s.valid)return;
            s=WeightSlot{};s.valid=true;s.dma=true;s.seq=nextPrefetch_++;s.bytes.resize(8192);
            if(bridgeWeights_ && s.seq>=(nBase_+nCount_)*kCount_)++bridgePrefetches_;
            uint32_t n=s.seq/kCount_,k=s.seq%kCount_;if(!fetch_)hbmRead_+=8192;++prefetchRequests_;
            const uint64_t source=weight_(tiles_[bridgeWeights_?n:nBase_+n],kBase_+k),destination=bBase_+uint64_t(index)*8192;
            const auto done=[this,index](bool ok){if(!ok){finish(false);return;}slots_[index].dma=false;slots_[index].reading=true;};
            if(fetch_)fetch_(source,destination,done);
            else gm_->dma_read_from_host_to_globalmem(source,8192,destination,done);
        }
    }
    void pumpWeight(uint32_t index){auto& s=slots_[index];if(!s.valid || !s.reading || s.pending)return;
        const size_t len=std::min(gm_->localMaxRequestBytes(),8192-s.offset);s.pending=true;
        if(!gm_->localReadAsync(bBase_+uint64_t(index)*8192+s.offset,len,LocalMemoryClient::RoCC,++tag_,
            [this,index,len](bool ok,uint64_t,const std::vector<uint8_t>& b){auto& s=slots_[index];s.pending=false;
                if(!ok || b.size()!=len){finish(false);return;}std::copy(b.begin(),b.end(),s.bytes.begin()+s.offset);
                s.offset+=len;localRead_+=len;if(s.offset==8192){s.reading=false;s.ready=true;}}))s.pending=false;
    }
    void loadA(){phase_=0;pending_=true;input_(kBase_+loadedA_,[this](bool ok,const Values& a){pending_=false;
        if(!ok || a.size()!=rows_*64){finish(false);return;}
        write(aBase_+uint64_t(loadedA_)*rows_*128,pack(a,2),[this]{
            if(++loadedA_<kCount_)schedule([this]{loadA();});else schedule([this]{restoreC();});});});
    }
    bool acquireArrays(){
        if(acquire_ && !leaseHeld_){
            if(earlyA_)prepareInput(n_*kCount_+k_);
            if(!acquire_()){++leaseWait_;return false;}
            leaseHeld_=true;
        }
        return true;
    }
    void restoreC(){
        phase_=1;
        // C/A/B reads do not touch the physical array context. Do them before
        // leasing it; programC/programB are the first destructive operations.
        if(lateLease_ && acquire_ && earlyA_)prepareInput(n_*kCount_+k_);
        if(!lateLease_ && !acquireArrays()){schedule([this]{restoreC();});return;}
        if(remaining_[0] || remaining_[1]){++waitArray_;schedule([this]{restoreC();});return;}
        if(kBase_){cRead_+=uint64_t(rows_)*256;read(cBase_+uint64_t(n_)*rows_*256,rows_*256,
            [this](const std::vector<uint8_t>& b){result_=unpack(b,4);schedule([this]{programC();});});}
        else if(initial_){pending_=true;initial_(tiles_[nBase_+n_],[this](bool ok,const Values& v){pending_=false;
            if(!ok || v.size()!=rows_*64){finish(false);return;}result_=v;schedule([this]{programC();});});}
        else schedule([this]{acquireTile();});
    }
    void programC(){
        phase_=1;
        if(cycle_<cReadyCycle_ || !acquireArrays()){schedule([this]{programC();});return;}pending_=true;
        if(!wcp_->writeGemmOutputGroupClassAsync(arrays_,result_,4,AttentionClusterTrafficClass::ProjectionOutput,++tag_,cycle_,
            [this](bool ok,uint64_t){pending_=false;if(!ok){finish(false);return;}++restores_;result_.clear();schedule([this]{acquireTile();});})){
            pending_=false;schedule([this]{programC();});}
    }
    void acquireTile(){
        phase_=2;
        const uint32_t seq=currentSequence();auto& slot=slots_[seq%slotCount_];
        if(earlyA_)prepareInput(seq);
        if(!slot.ready || slot.seq!=seq){if(slot.dma)++waitDma_;else ++waitGm_;schedule([this]{acquireTile();});return;}
        ++prefetchHits_;matrix_=unpack(slot.bytes,2);group_=0;
        prepareInput(seq);
        schedule([this]{programB();});
    }
    void prepareInput(uint32_t seq){
        if(preparedSeq_==seq)return;preparedSeq_=seq;
        if(inputPrefetch_){
            auto& current=inputSlots_[seq%2];
            if(current.valid && current.seq==seq && current.ready)++inputHits_;
            for(uint32_t next=seq;next<std::min(seq+2,sequenceLimit());++next){
                auto& s=inputSlots_[next%2];
                if(s.valid){if(s.seq!=next){finish(false);return;}continue;}
                s=InputSlot{};s.valid=true;s.seq=next;
                const uint32_t k=next%kCount_;
                s.address=inputAddress_?inputAddress_(kBase_+k):aBase_+uint64_t(k)*rows_*128;
                s.bytes.resize(rows_*128);
            }
            uint64_t live=0;for(const auto& s:inputSlots_)if(s.valid)++live;
            inputHighWater_=std::max(inputHighWater_,live);return;
        }
        inputOffset_=0;inputPending_=inputReady_=false;inputReading_=true;inputBytes_.assign(rows_*128,0);
        inputReadAddress_=inputAddress_?inputAddress_(kBase_+k_):aBase_+uint64_t(k_)*rows_*128;
    }
    void pumpInput(){
        if(inputPrefetch_){
            // Two bounded snapshots, each backed by an existing immutable GM
            // input tile. All bytes still cross the timed local port. Neither
            // snapshot programs an array before its preceding MAC completes.
            for(uint32_t i=0;i<2;++i)pumpInputSlot((preparedSeq_+i)%2);
            return;
        }
        if(!inputReading_ || inputPending_)return;
        const size_t len=std::min(gm_->localMaxRequestBytes(),inputBytes_.size()-inputOffset_);inputPending_=true;
        if(!gm_->localReadAsync(inputReadAddress_+inputOffset_,len,LocalMemoryClient::RoCC,++tag_,
            [this,len](bool ok,uint64_t,const std::vector<uint8_t>& b){inputPending_=false;
                if(!ok || b.size()!=len){finish(false);return;}std::copy(b.begin(),b.end(),inputBytes_.begin()+inputOffset_);
                inputOffset_+=len;localRead_+=len;if(inputOffset_==inputBytes_.size()){
                    inputReading_=false;inputReady_=true;inputValues_=unpack(inputBytes_,2);inputBytes_.clear();}}))inputPending_=false;
    }
    void pumpInputSlot(uint32_t index){
        auto& s=inputSlots_[index];if(!s.valid || s.ready || s.pending)return;
        const size_t len=std::min(gm_->localMaxRequestBytes(),s.bytes.size()-s.offset);s.pending=true;
        if(!gm_->localReadAsync(s.address+s.offset,len,LocalMemoryClient::RoCC,++tag_,
            [this,index,len](bool ok,uint64_t,const std::vector<uint8_t>& b){auto& s=inputSlots_[index];s.pending=false;
                if(!ok || b.size()!=len){finish(false);return;}
                std::copy(b.begin(),b.end(),s.bytes.begin()+s.offset);s.offset+=len;localRead_+=len;
                if(s.offset==s.bytes.size()){s.ready=true;s.values=unpack(s.bytes,2);s.bytes.clear();}
            }))s.pending=false;
    }
    std::vector<uint32_t> groupArrays() const {
        uint32_t begin=groups_==2?group_*32:0,end=groups_==2?std::min(begin+32,rows_):rows_;
        return std::vector<uint32_t>(arrays_.begin()+begin,arrays_.begin()+end);
    }
    void programB(){
        phase_=3;
        if(!acquireArrays()){schedule([this]{programB();});return;}
        if(remaining_[group_] || (scatterWave_ && remaining_[1])){++waitArray_;schedule([this]{programB();});return;}pending_=true;
        const uint64_t submitted=cycle_;
        // One broadcast to the existing 64 arrays, followed by two scatter/MAC
        // waves. No second matrix programming pass or operand context.
        if(!wcp_->programGemmMatrixGroupClassBankAsync(scatterWave_?arrays_:groupArrays(),0,matrix_,2,AttentionClusterTrafficClass::ProjectionWeights,++tag_,cycle_,
            [this,submitted](bool ok,uint64_t){pending_=false;if(!ok){finish(false);return;}groupProgram_+=cycle_-submitted;schedule([this]{programA();});})){
            pending_=false;schedule([this]{programB();});}
    }
    void programA(){
        phase_=4;
        if(!(inputPrefetch_?inputSlots_[preparedSeq_%2].ready:inputReady_)){++waitA_;schedule([this]{programA();});return;}
        uint32_t begin=groups_==2?group_*32:0,end=groups_==2?std::min(begin+32,rows_):rows_;
        const auto& input=inputPrefetch_?inputSlots_[preparedSeq_%2].values:inputValues_;
        Values a(input.begin()+begin*64,input.begin()+end*64);pending_=true;
        if(!wcp_->programGemmInputScatterBankAsync(groupArrays(),0,a,2,AttentionClusterTrafficClass::ProjectionInputScatter,++tag_,cycle_,
            [this](bool ok,uint64_t){pending_=false;if(!ok){finish(false);return;}schedule([this]{launch();});})){
            pending_=false;schedule([this]{programA();});}
    }
    void launch(){
        phase_=5;
        auto ids=groupArrays();uint32_t group=group_,rows=ids.size();remaining_[group]=rows;const uint64_t launched=cycle_;
        if(!wcp_->launchGemmArrayGroupActiveBank(ids,0,(kBase_+k_ || initial_)?1:0,64,cycle_,
            [this,group,rows,launched](uint32_t,uint64_t){if(!--remaining_[group])groupCompute_+=(cycle_-launched)*rows;})){
            remaining_[group]=0;schedule([this]{launch();});return;}
        ++groupLaunches_;
        if(++group_<groups_){schedule([this]{if(scatterWave_)programA();else programB();});return;}
        ++launches_;matrix_.clear();inputValues_.clear();
        if(inputPrefetch_)inputSlots_[preparedSeq_%2]=InputSlot{};
        slots_[currentSequence()%slotCount_]=WeightSlot{};fillPrefetch();
        if(++k_<kCount_)schedule([this]{acquireTile();});else schedule([this]{drain();});
    }
    void drain(){
        phase_=6;
        if(remaining_[0] || remaining_[1]){++waitArray_;schedule([this]{drain();});return;}
        if(outputWaiting_){++waitOutput_;schedule([this]{drain();});return;}
        pending_=true;
        const uint32_t bytes=kBase_+kCount_<kTiles_?4:2;
        if(!wcp_->readGemmOutputGroupClassAsync(arrays_,bytes,AttentionClusterTrafficClass::ProjectionOutput,++tag_,cycle_,
            [this](bool ok,uint64_t,const Values& v){pending_=false;if(!ok || v.size()!=rows_*64){finish(false);return;}++drains_;
                // C is now in the acknowledged read snapshot; no array state
                // remains live until restoreC acquires the next tile lease.
                if(leaseHeld_){leaseHeld_=false;release_();}
                if(kBase_+kCount_<kTiles_){auto bytes=pack(v,4);cWrite_+=bytes.size();
                    write(cBase_+uint64_t(n_)*rows_*256,std::move(bytes),[this]{advanceN();});}
                else{result_=v;outputWaiting_=true;outputAdvanced_=detachOutput_;phase_=7;
                    output_(tiles_[nBase_+n_],result_);
                    if(detachOutput_){result_.clear();advanceN();}}})){
            pending_=false;schedule([this]{drain();});}
    }
    void advanceN(){
        k_=0;if(++n_<nCount_){schedule([this]{restoreC();});return;}
        kBase_+=kCount_;
        if(kBase_>=kTiles_){nBase_+=nCount_;kBase_=0;if(nBase_>=tiles_.size()){waitFinish();return;}}
        schedule([this]{beginWindow();});
    }
    void waitFinish(){phase_=8;if(outputWaiting_){++waitOutput_;schedule([this]{waitFinish();});return;}finish(true);}
    GlobalMemoryAPI* gm_;WorkerCommandProcessorAPI* wcp_;
    bool active_=false,pending_=false,outputWaiting_=false;
    bool earlyA_=false,lateLease_=false,detachOutput_=false,outputAdvanced_=false,aFirst_=false;
    uint32_t gmLookahead_=4;uint64_t gmDeferred_=0;
    uint32_t currentSequence() const {return (bridgeWeights_?nBase_*kCount_:0)+n_*kCount_+k_;}
    uint32_t sequenceLimit() const {return (bridgeWeights_?uint32_t(tiles_.size()):nCount_)*kCount_;}
    bool inputPrefetch_=false,scatterWave_=false,bridgeWeights_=false;std::array<InputSlot,2> inputSlots_{};
    uint64_t inputHighWater_=0,inputHits_=0;
    uint32_t preparedSeq_=~uint32_t(0),phase_=0;
    std::array<uint64_t,9> phases_{};
    uint64_t activeCycles_=0,waitDma_=0,waitGm_=0,waitA_=0,waitArray_=0,waitOutput_=0;
    uint64_t outputPendingCycles_=0,outputComputeOverlap_=0;
    uint32_t rows_=0,kTiles_=0,nBase_=0,kBase_=0,n_=0,k_=0,nCount_=0,kCount_=0,loadedA_=0;
    uint32_t kWindow_=32,slotCount_=4,nextPrefetch_=0,groups_=1,group_=0;
    std::array<uint32_t,2> remaining_{};std::array<WeightSlot,4> slots_{};
    uint64_t cycle_=0,tag_=0,aBase_=0,bBase_=0,cBase_=0,cReadyCycle_=0,ioAddr_=0;
    uint64_t localRead_=0,localWrite_=0,hbmRead_=0,windows_=0,launches_=0;
    uint64_t prefetchRequests_=0,prefetchHits_=0,cRead_=0,cWrite_=0,drains_=0,restores_=0,bridgePrefetches_=0;
    uint64_t groupLaunches_=0,groupOverlap_=0,transferOverlap_=0,groupProgram_=0,groupCompute_=0;
    size_t offset_=0;std::vector<uint32_t> arrays_,tiles_;std::vector<uint8_t> io_;
    Values matrix_,inputValues_,result_;
    Input input_;Initial initial_;Output output_;std::function<uint64_t(uint32_t,uint32_t)> weight_;
    std::function<uint64_t(uint32_t)> inputAddress_;
    Fetch fetch_;Acquire acquire_;Release release_;bool leaseHeld_=false;uint64_t leaseWait_=0;
    bool inputReading_=false,inputPending_=false,inputReady_=false;
    uint64_t inputReadAddress_=0;size_t inputOffset_=0;std::vector<uint8_t> inputBytes_;
    std::function<void(bool)> done_;std::function<void()> next_,writeDone_;
    std::function<void(const std::vector<uint8_t>&)> readDone_;
};
}}
#endif
