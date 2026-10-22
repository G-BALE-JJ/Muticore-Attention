#ifndef SST_GOLEM_DECODER_OP_JOB_H
#define SST_GOLEM_DECODER_OP_JOB_H
#include <sst/elements/golem/attention/decoderOpAbi.h>
#include <sst/elements/golem/attention/decoderMemoryLayout.h>
#include <sst/elements/golem/globalmemory/globalmemory.h>
#include <sst/elements/golem/workercmdproc/workercmdproc.h>
#include <sst/elements/golem/fp16.h>
#include <sst/elements/golem/attention/reuseWindowGemm.h>
#include <sst/elements/golem/attention/reuseWeightInterface.h>
#include <sst/elements/golem/attention/softwareLayerResidency.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <vector>
namespace SST { namespace Golem {
// One operand context and bounded, timed local scratch. Output panels may
// drain while subsequent tiles compute; completion waits for every HBM ACK.
class DecoderOpJob {
public:
    DecoderOpJob(GlobalMemoryAPI* gm, WorkerCommandProcessorAPI* wcp, SFUAPI* sfu,ReuseWeightInterface* sharing=nullptr)
        : gm_(gm), wcp_(wcp), sfu_(sfu), window_(gm,wcp), downWindow_(gm,wcp), sharing_(sharing) {
        allGemmWindow_=option("GOLEM_ALL_GEMM_REUSE_WINDOW",0,0,1);
        blockPipeline_=option("GOLEM_DECODER_FFN_BLOCK_PIPELINE",0,0,1);
        pipelinePolicy_=option("GOLEM_DECODER_FFN_PIPELINE_POLICY",2,0,2);
        reuseGamma_=option("GOLEM_DECODER_GAMMA_REUSE",0,0,1);
        streamFfn_=option("GOLEM_DECODER_FFN_SFU_STREAM",0,0,1);
        ffnChunkLimit_=option("GOLEM_DECODER_FFN_SFU_CHUNK",8192,1024,8192);
        normHandoff_=option("GOLEM_DECODER_NORM_FFN_HANDOFF",0,0,1);
        stageChain_=option("GOLEM_DECODER_GM_STAGE_CHAIN",0,0,1);
        addNormFusion_=option("GOLEM_DECODER_ADD_NORM_FUSION",0,0,1);
        layerResidency_=option("GOLEM_DECODER_LAYER_RESIDENCY",0,0,1);
        inputResidency_=option("GOLEM_DECODER_INPUT_RESIDENCY",0,0,1);
        lmInputResidency_=option("GOLEM_DECODER_LM_INPUT_RESIDENCY",0,0,1);
        elideIntermediateOutput_=option("GOLEM_DECODER_ELIDE_INTERMEDIATE_OUTPUT",0,0,1);
        for (uint32_t i = 0; i < 64; ++i) arrays_.push_back(i);
        panelLimit_=option("GOLEM_DECODER_OUTPUT_PANEL_TILES",16,0,16);
        writeWindow_=option("GOLEM_DECODER_OUTPUT_WRITE_WINDOW",8,1,16);
        prefetchEnabled_=option("GOLEM_DECODER_GEMM_PREFETCH",0,0,1);
        prefetchSlots_=option("GOLEM_DECODER_PREFETCH_SLOTS",2,2,8);
        if (prefetchSlots_!=2 && prefetchSlots_!=4 && prefetchSlots_!=8)
            throw std::invalid_argument("GOLEM_DECODER_PREFETCH_SLOTS");
        weightTrace_=option("GOLEM_DECODER_WEIGHT_TRACE",0,0,1);
        ffnPanelLimit_=option("GOLEM_DECODER_FFN_PANEL_TILES",16,1,16);
        ffnLayout_=option("GOLEM_DECODER_FFN_LAYOUT",0,0,2);
        downInputReuse_=option("GOLEM_DECODER_DOWN_INPUT_REUSE",0,0,1);
        arrayGroups_=option("GOLEM_DECODER_ARRAY_GROUPS",1,1,2);
        lmParallel_=option("GOLEM_DECODER_LM_N_PARALLEL",1,1,8);
        if (lmParallel_!=1 && lmParallel_!=2 && lmParallel_!=4 && lmParallel_!=8)
            throw std::invalid_argument("GOLEM_DECODER_LM_N_PARALLEL");
    }
    bool start(const DecoderOpDesc& d) {
        if (active_ || done_ || !gm_ || !wcp_ || !sfu_ || wcp_->isBusy() ||
            d.magic != GOLEM_DECODER_OP_MAGIC || d.size_bytes != sizeof(d) ||
            d.op < 1 || d.op > 9 || !d.rows ||
            d.lane > 3 || d.manager > 3 || !d.input_dim || (d.op!=6 && d.input_dim % 64) ||
            !d.output_dim || d.output_dim % 64 || (d.op!=6 && d.input_dim > 8192) || (d.op==6 && d.input_dim>1048576) ||
            d.output_dim > 8192 || !d.input_addr || !d.output_addr ||
            (d.op != 1 && d.op!=6 && d.op!=8 && d.op!=9 && d.input_dim != d.output_dim) ||
            (d.op != 1 && d.op!=9 && d.output_tile_begin) || (d.rows==1 && d.op==1 && d.head_dim) ||
            ((d.op == 2 || d.op == 3 || d.op == 5 || d.op==7 || d.op==8 || d.op==9) && !d.input1_addr) ||
            (d.op==8 && d.head_dim) ||
            ((d.op == 1 || d.op==6 || d.op==8 || d.op==9) && (!d.weights_offset || !d.node_stride ||
                (d.op!=9 && d.head_dim && (d.head_dim % 64 || d.input_dim % d.head_dim)))) ||
            gm_->getSize() < DecoderMemoryLayout::decoder_required_gm ||
            (d.op==9 && (!d.down_weights_offset || d.input_dim>2048 ||
                gm_->getSize()<DecoderMemoryLayout::decoder_down_required_gm))) return false;
        chainJob_=0;
        if(stageChain_ && chainState_) {
            const uint32_t expected=chainState_==1?2:chainState_==2?3:4;
            const bool same=d.epoch==chainDesc_.epoch && d.manager==chainDesc_.manager &&
                d.lane==chainDesc_.lane && d.rows==chainDesc_.rows && d.input_dim==chainDesc_.input_dim &&
                d.node_stride==chainDesc_.node_stride && (expected==4 || d.output_dim==d.input_dim);
            const bool match=expected==2?(d.op==2 && d.stage==2 && d.input_addr==chainDesc_.output_addr):
                expected==3?(d.op==3 && d.stage==3 && d.input_addr==chainH1Address_):
                (d.op==9 && d.stage==4 && d.input_addr==chainNormAddress_ &&
                 d.residual_offset && (d.manager+1)*d.node_stride+d.residual_offset==chainH1Address_);
            // A previous producer deliberately omitted HBM materialization.
            // A mismatched consumer must never fall back to stale HBM.
            if(!same || !match)return false;
            chainJob_=expected;
        } else if(stageChain_ && allGemmWindow_ && d.op==1 && d.stage==1 &&
                  d.rows<=256 && d.lane*64<d.rows && d.input_dim<=2048 &&
                  d.output_dim==d.input_dim && panelLimit_ && ffnLayout_==2 &&
                  !normHandoff_ && !blockPipeline_ && !sharing_) {
            chainJob_=1;chainDesc_=d;
        }
        fusedAddNorm_=addNormFusion_ && chainJob_==2;
        // Software descriptor extension, using fields unused by residual Add.
        // Refuse an incomplete fused consumer instead of reading stale HBM.
        if(fusedAddNorm_ && (!d.weights_offset || !d.head_dim || !d.node_stride || d.completion_addr))return false;
        if(d.op==2 && d.stage==2 && d.head_dim && !fusedAddNorm_)return false;
        fusedNormPhase_=false;fusedNormChunks_=fusedNormReadSaved_=0;
        chainReadSaved_=chainWriteSaved_=chainLocalBytes_=0;
        layerProducer_=layerResidency_ && chainJob_==4;
        const uint32_t layers=configuredLayers();
        elideOutput_=elideIntermediateOutput_ && inputResidency_ && layerProducer_ && stageChain_ &&
            allGemmWindow_ && ffnLayout_==2 && !normHandoff_ && !blockPipeline_ &&
            !sharing_ && layers>1 && d.epoch>0 && d.epoch<layers;
        elidedOutputBytes_=0;
        layerConsumer_=layerResidency_ && stageChain_ && d.op==3 && d.stage==10 &&
            d.rows<=256 && d.input_dim<=2048 && d.lane*64<d.rows;
        layerResidualConsumer_=inputResidency_ && layerResidency_ && stageChain_ &&
            chainJob_==2 && d.op==2 && d.stage==2 && d.rows<=256 &&
            d.input_dim<=2048 && d.lane*64<d.rows;
        layerPublished_=false;layerBytes_=layerSaved_=0;
        layerResidualSaved_=0;
        useLmInput_=lmInputResidency_ && lmInputValid_ && d.op==1 && d.stage==11 && d.rows==1 &&
            d.input_addr==lmInputDesc_.input_addr && d.input_dim==lmInputDesc_.input_dim &&
            d.manager==lmInputDesc_.manager && d.lane==lmInputDesc_.lane &&
            d.epoch==lmInputDesc_.epoch+1 && d.output_tile_begin==lmInputDesc_.output_tile_begin+lmInputDesc_.output_dim/64;
        lmInputValid_=false;lmInputSaved_=0;
        residualCacheNext_=0;
        if (d.op==9 && !wcp_->beginProjectionPartialBuffer(64ull*d.input_dim*2)) return false;
        useNormCache_=normHandoff_ && allGemmWindow_ && d.op==9 && normCacheValid_ &&
            d.rows==normCacheDesc_.rows && d.input_dim==normCacheDesc_.input_dim &&
            d.lane==normCacheDesc_.lane && d.manager==normCacheDesc_.manager &&
            d.epoch==normCacheDesc_.epoch && d.input_addr==normCacheDesc_.output_addr;
        normCacheValid_=false;
        cacheNorm_=normHandoff_ && allGemmWindow_ && d.op==3 && d.stage==3 &&
            d.rows<=256 && d.input_dim<=2048 && d.input_dim%64==0 && d.lane*64<d.rows;
        if(cacheNorm_)normCacheDesc_=d;
        normCacheWritten_=normCacheReadSaved_=0;
        d_ = d; singleRow_=d.rows==1 && d.op==1;
        sharedWeights_=allGemmWindow_ && sharing_ && d.op==9;sharedAccounted_=false;
        if(sharedWeights_)sharing_->begin(d.epoch);
        if(blockPipeline_ && d.op==9 && (!allGemmWindow_ || ffnLayout_!=2 || sharedWeights_))
            throw std::invalid_argument("FFN block pipeline requires private all-window layout2");
        pipelineProducerBlock_=false;
        arrayOwner_=0;arrayPreferred_=1;arrayWaiting_[0]=arrayWaiting_[1]=false;
        downTask_={};pipelineWrite_.clear();pipelineWritePending_=false;pipelineProducerTransfers_=pipelineDownTransfers_=pipelineBothActive_=0;pipelinePasses_=0;
        pipelineLeaseWait_=0;
        gammaLoaded_=false;
        row_ = singleRow_ ? 0 : d.lane * 64; nt_ = kt_ = 0; active_ = true;
        status_ = 0; started_ = false; hbmRead_ = hbmWrite_ = localRead_ = localWrite_ = 0;
        panels_[0]=Panel{}; panels_[1]=Panel{}; panelIndex_=0; panelCount_=0;
        outputRequests_=outputHighWater_=0;
        for (uint32_t i=0;i<DecoderMemoryLayout::decoder_prefetch_slots;++i) weightReads_[i]=BackgroundRead{};
        inputRead_=BackgroundRead{}; weightSlot_=0; downWeightSlot_=0;
        prefetchRequests_=prefetchHits_=prefetchWaitCycles_=0;
        prefetchDmaWaitCycles_=prefetchGmWaitCycles_=prefetchProgramWaitCycles_=arrayWaitCycles_=0;
        weightDmaCycles_=weightGmCycles_=weightProgramCycles_=arrayComputeCycles_=0;
        groupRemaining_[0]=groupRemaining_[1]=0;
        groupLaunches_=groupProgramCycles_=groupComputeArrayCycles_=groupOverlapCycles_=0;
        lmLaunches_=lmArrayTiles_=0;
        siluMultiplySecond_=false;
        ffnUp_=false; ffnReady_[0]=ffnReady_[1]=false;
        ffnState_=FfnState::Idle; ffnConsumer_=0; ffnOffset_=0;
        ffnGateRead_=BackgroundRead{}; ffnUpRead_=BackgroundRead{};
        ffnWrite_.clear(); ffnWritePending_=false;
        for(auto& chunk:ffnStream_)chunk=FfnChunk{};
        ffnStreamOverlap_=ffnStreamHighWater_=0;
        ffnProduced_=ffnConsumed_=ffnChunks_=ffnOverlapCycles_=ffnSlotWaitCycles_=0;
        downActive_=false; downNextK_=0; downPartialReady_=0;
        downReady_[0]=downReady_[1]=false; downPanels_[0]=Panel{}; downPanels_[1]=Panel{};
        downOutputIndex_=downOutputCount_=0; downCRead_=downCWrite_=downRestores_=downPasses_=0;
        downOwnedBuffer_=d.op==9;
        residualState_=ResidualState::Idle;residualRead_=BackgroundRead{};residualDownRead_=BackgroundRead{};
        residualPending_=0;residualWritePending_=false;residualWrite_.clear();residualChunks_=residualBytes_=residualOverlap_=0;
        stridedPending_=stridedNext_=stridedDone_=0;
        defer([this] {
            if(!layerConsumer_ && !layerResidualConsumer_){loadRows();return;}
            readLocal(gm_->getBaseAddr()+SoftwareLayerResidency::record,64,[this](const std::vector<uint8_t>& b){
                const uint64_t expected=layerResidualConsumer_?d_.input1_addr:d_.input_addr;
                if(!SoftwareLayerResidency::matches(b,d_.epoch,expected,d_.rows,d_.input_dim,d_.lane,d_.manager)){
                    finish(false);return;
                }
                loadRows();
            });
        });
        return true;
    }
    bool active() const { return active_; }
    bool complete() const { return done_; }
    uint64_t status() const { return status_; }
    void retire() { if (done_) { done_ = false; next_ = {}; } }
    void tick(uint64_t cycle, uint64_t tick) {
        cycle_ = cycle; tick_ = tick;
        if (!active_) return;
        if (groupRemaining_[0] && groupRemaining_[1]) ++groupOverlapCycles_;
        window_.tick(cycle);
        if(pipelined()){
            downWindow_.tick(cycle);
            if(downTask_){auto f=std::move(downTask_);downTask_={};f();}
            pumpPipelineWrite();
            if(window_.active() && downWindow_.active())++pipelineBothActive_;
            if(window_.computing() && downWindow_.transferring())++pipelineDownTransfers_;
            if(downWindow_.computing() && window_.transferring())++pipelineProducerTransfers_;
        }
        pumpOutputs();
        for (auto& read:weightReads_) pumpRead(read);
        pumpRead(inputRead_);
        pumpRead(ffnGateRead_); pumpRead(ffnUpRead_);
        for (auto& read:downWeightReads_) pumpRead(read);
        pumpRead(downInputRead_);
        if (isFfn()) pumpFfn();
        if(pipelined())pumpPipelineDown();
        if (d_.op==9 && d_.residual_offset) pumpResidual();
        if (!started_) {
            started_ = true; startTick_ = tick;
            std::printf("[DECODER_OP] event=start core=%u stage=%u op=%u tick=%llu K=%u N=%u epoch=%u\n",
                core(), d_.stage, d_.op, (unsigned long long)tick, d_.input_dim, d_.output_dim, d_.epoch);
        }
        if (vectorWaiting_ && vectorResult_.completed && tick >= vectorResult_.readyTick) {
            vectorWaiting_ = false;
            if(fusedAddNorm_)std::printf("[DECODER_ADD_NORM_SFU] core=%u epoch=%u op=%u elements=%zu submit_tick=%llu done_tick=%llu\n",
                core(),d_.epoch,fusedNormPhase_?2u:3u,vectorResult_.values.size(),
                (unsigned long long)fusedVectorSubmit_,(unsigned long long)vectorResult_.readyTick);
            if (d_.op==7 && !siluMultiplySecond_) {
                // SFU SiLU returns FP16-rounded values. Preserve that boundary
                // while retaining its bounded 8192-element result on chip.
                vectorInput_=std::move(vectorResult_.values);
                siluMultiplySecond_=true; defer([this] { issueVector(); });
            } else {
            auto bytes = encode(vectorResult_.values);
            vectorResult_.values.clear();
            if(fusedAddNorm_ && fusedNormPhase_) {
                ++fusedNormChunks_;cacheNormChunk(bytes);
            } else if(chainJob_==2) {
                const size_t count=bytes.size();
                if(fusedAddNorm_)vectorInput_=decode(bytes); // Preserve the Add FP16 rounding boundary.
                writeLocal(chainGM()+uint64_t(vectorRow_-row_)*d_.input_dim*2,bytes,[this,count]{
                    chainLocalBytes_+=count;chainWriteSaved_+=count;
                    if(fusedAddNorm_)loadFusedNormGamma();else advanceVectorRows();});
            } else if(chainJob_==3)cacheNormChunk(bytes);
            else if(cacheNorm_)writeLocal(outGM(),bytes,[this,bytes]{cacheNormChunk(bytes);});
            else writeLocal(outGM(), bytes, [this] { storeVector(); });
            }
        }
        if (next_) { auto f = std::move(next_); next_ = {}; f(); }
    }
private:
    bool tileMajorFfn() const { return d_.op==9 && ffnLayout_; }
    bool reuseDownInput() const { return d_.op==9 && downInputReuse_; }
    uint64_t downInputGM() const { return tileMajorFfn()?panelGM(downConsumer_):upPanelGM(downConsumer_); }
    bool isFfn() const { return d_.op==8 || d_.op==9; }

    bool pipelined() const {return blockPipeline_ && d_.op==9;}
    bool pipelineProducerSlotAvailable() const {
        const bool consuming=ffnState_!=FfnState::Idle && ffnConsumer_==panelIndex_;
        return !panels_[panelIndex_].draining && !ffnReady_[panelIndex_] && !downReady_[panelIndex_] &&
               !consuming && !(downActive_ && downConsumer_==panelIndex_);
    }
    bool acquirePipelineArrays(uint32_t owner) {
        arrayWaiting_[owner-1]=true;
        if(arrayOwner_)return false;
        const bool producerReady=pipelineProducerSlotAvailable();
        if(owner==1 && !producerReady){++ffnSlotWaitCycles_;return false;}
        // Finish a Gate/Up pair before lending arrays to Down, so its SFU
        // work can cover the following consumer pass. Prefetch is independent.
        if(pipelinePolicy_ && pipelineProducerBlock_ && producerReady){if(owner==2)return false;}
        else if(arrayPreferred_!=owner && arrayWaiting_[2-owner] && (owner==1 || producerReady))return false;
        arrayOwner_=owner;arrayWaiting_[owner-1]=false;return true;
    }
    void releasePipelineArrays(uint32_t owner) {
        if(arrayOwner_!=owner)throw std::logic_error("FFN array lease owner");
        arrayOwner_=0;arrayPreferred_=3-owner;
    }
    void accountWindow(const char* name) {accountWindow(name,window_);}
    void accountWindow(const char* name,ReuseWindowGemm& window_) {
        if(pipelined())pipelineLeaseWait_+=window_.leaseWaitCycles();
        window_.trace(core(),d_.epoch,d_.stage,name);
        localRead_+=window_.localReadBytes(); localWrite_+=window_.localWriteBytes();hbmRead_+=window_.hbmReadBytes();
        prefetchRequests_+=window_.prefetchRequests();prefetchHits_+=window_.prefetchHits();
        groupLaunches_+=window_.groupLaunches();groupOverlapCycles_+=window_.groupOverlapCycles();
        groupProgramCycles_+=window_.groupProgramCycles();groupComputeArrayCycles_+=window_.groupComputeArrayCycles();
        std::printf("[GEMM_REUSE_WINDOW] core=%u stage=%u epoch=%u operator=%s row=%u reuse_n=4 reuse_m=1 k_window=%u windows=%llu launches=%llu hbm_read=%llu gm_read=%llu gm_write=%llu c_gm_read=%llu c_gm_write=%llu c_drains=%llu c_restores=%llu transfer_compute_overlap=%llu status=0\n",
            core(),d_.stage,d_.epoch,name,row_,window_.kWindow(),(unsigned long long)window_.windows(),
            (unsigned long long)window_.launches(),(unsigned long long)window_.hbmReadBytes(),
            (unsigned long long)window_.localReadBytes(),(unsigned long long)window_.localWriteBytes(),
            (unsigned long long)window_.cReadBytes(),(unsigned long long)window_.cWriteBytes(),
            (unsigned long long)window_.drains(),(unsigned long long)window_.restores(),
            (unsigned long long)window_.transferOverlapCycles());
    }
    void startProducerWindow() {
        if(pipelined() && !ffnUp_)pipelineProducerBlock_=true;
        std::vector<uint32_t> tiles;
        const uint32_t first=nt_;
        const uint32_t limit=panelEnabled()?std::min(d_.output_dim/64,nt_+panelTiles()):d_.output_dim/64;
        for(uint32_t n=nt_;n<limit;n+=singleRow_?16:1)tiles.push_back(n);
        const uint32_t after=singleRow_?d_.output_dim/64:limit;
        const uint64_t local=gm_->getBaseAddr();
        const char* name=isFfn()?(ffnUp_?"up":"gate"):(singleRow_?"lm_head":"linear");
        window_.start(activeRows_,d_.input_dim/64,std::move(tiles),
            inputGM(),
            local+(pipelined()?DecoderMemoryLayout::decoder_prefetch_gm:DecoderMemoryLayout::decoder_down_weight_gm),
            local+(pipelined()?DecoderMemoryLayout::decoder_pipeline_producer_c_gm:DecoderMemoryLayout::decoder_prefetch_gm),
            [this](uint32_t k,ReuseWindowGemm::Loaded done){
                readLocal(inputGM()+uint64_t(k)*activeRows_*128,activeRows_*128,
                    [done](const std::vector<uint8_t>& b){done(true,decode(b));});},
            [this](uint32_t n,uint32_t k){return weightAddress(n,k,ffnUp_);},
            [this,first](uint32_t n,const std::vector<double>& values){
                nt_=n;
                if(chainJob_==1) {
                    const size_t bytes=values.size()*2;
                    writePanelTile(chainWoGM(),d_.output_dim/64,n,encode(values),[this,bytes]{
                        chainLocalBytes_+=bytes;chainWriteSaved_+=bytes;++panelCount_;window_.resumeOutput();});
                } else if(panelEnabled()) {
                    writeLocal(producerPanelGM()+uint64_t(n-first)*activeRows_*128,encode(values),
                        [this]{++panelCount_;window_.resumeOutput();});
                } else {
                    // LM owns strided N tiles; retain its original HBM placement.
                    writeLocal(outGM(),encode(values),[this]{
                        storeHBM(outGM(),d_.output_addr+uint64_t(nt_)*128,128,[this]{window_.resumeOutput();});});
                }
            },
            [this,after,name](bool ok){if(!ok){finish(false);return;}accountWindow(name);nt_=after;kt_=0;
                if(chainJob_==1){panelCount_=0;if(nt_<d_.output_dim/64)defer([this]{loadWeight();});else{row_+=256;defer([this]{loadRows();});}}
                else if(panelEnabled())defer([this]{preparePanel();});
                else{row_+=256;defer([this]{loadRows();});}}, {},
            [this](uint32_t k){return inputGM()+uint64_t(k)*activeRows_*128;},true,weightFetcher(),
            pipelined()?ReuseWindowGemm::Acquire([this]{return acquirePipelineArrays(1);}):ReuseWindowGemm::Acquire{},
            pipelined()?ReuseWindowGemm::Release([this]{releasePipelineArrays(1);}):ReuseWindowGemm::Release{});
    }
    void initialDownWindow(uint32_t n,ReuseWindowGemm::Loaded done) {
        if(cycle_<downPartialReady_) {defer([this,n,done]{initialDownWindow(n,done);});return;}
        std::vector<uint8_t> b;uint64_t ready=0;
        if(!wcp_->readProjectionPartial(uint64_t(n)*activeRows_*128,activeRows_*128,cycle_,b,ready)){
            done(false,{});return;}
        downCRead_+=b.size();
        waitInitialDown(ready,std::move(b),std::move(done));
    }
    void waitInitialDown(uint64_t ready,std::vector<uint8_t> b,ReuseWindowGemm::Loaded done) {
        if(cycle_<ready){defer([this,ready,b,done]{waitInitialDown(ready,b,done);});return;}
        ++downRestores_;done(true,decode(b));
    }
    void startDownWindow() {
        const bool final=panels_[downConsumer_].firstTile+panels_[downConsumer_].tiles==d_.output_dim/64;
        if(final && (downPanels_[downOutputIndex_].draining || downPanels_[downOutputIndex_].epilogue)){
            defer([this]{startDownWindow();});return;}
        std::vector<uint32_t> tiles;for(uint32_t n=0;n<d_.input_dim/64;++n)tiles.push_back(n);
        ReuseWindowGemm::Initial initial;
        if(panels_[downConsumer_].firstTile)initial=[this](uint32_t n,ReuseWindowGemm::Loaded cb){initialDownWindow(n,std::move(cb));};
        const uint64_t local=gm_->getBaseAddr();
        window_.start(activeRows_,panels_[downConsumer_].tiles,std::move(tiles),
            downInputGM(),
            local+DecoderMemoryLayout::decoder_down_weight_gm,
            local+DecoderMemoryLayout::decoder_prefetch_gm,
            [this](uint32_t k,ReuseWindowGemm::Loaded done){readLocal(downInputGM()+uint64_t(k)*activeRows_*128,
                activeRows_*128,[done](const std::vector<uint8_t>& b){done(true,decode(b));});},
            [this](uint32_t n,uint32_t k){return downWeightAddress(n,k);},
            [this](uint32_t n,const std::vector<double>& v){downN_=n;downK_=0;windowDownSink_=true;consumeDownOutput(v);},
            [this](bool ok){if(!ok){finish(false);return;}accountWindow("down");downN_=d_.input_dim/64; windowDownSink_=false; downActive_=false;
                defer([this]{downNextOutput();});},std::move(initial),
            [this](uint32_t k){return downInputGM()+uint64_t(k)*activeRows_*128;},false,weightFetcher());
    }
    // Bounded producer/consumer software pipeline. The two engines use
    // independent weight slots; all 64 arrays are leased for one C tile at a
    // time. Only acknowledged GM writes publish a gated panel to Down.
    void pumpPipelineDown() {
        if(downActive_ || downTask_ || !pipelineWrite_.empty())return;
        for(uint32_t i=0;i<2;++i){
            if(!downReady_[i] || panels_[i].firstTile!=downNextK_)continue;
            downReady_[i]=false;downActive_=true;downConsumer_=i;downN_=0;downK_=0;
            const auto& p=panels_[i];
            std::vector<uint32_t> tiles;for(uint32_t n=0;n<d_.input_dim/64;++n)tiles.push_back(n);
            ReuseWindowGemm::Initial initial;
            if(p.firstTile)initial=[this](uint32_t n,ReuseWindowGemm::Loaded cb){pipelineInitialDown(n,std::move(cb));};
            const uint64_t local=gm_->getBaseAddr();
            downWindow_.start(activeRows_,p.tiles,std::move(tiles),panelGM(i),
                local+DecoderMemoryLayout::decoder_down_weight_gm,
                local+DecoderMemoryLayout::decoder_pipeline_down_c_gm,
                {},[this](uint32_t n,uint32_t k){return downWeightAddress(n,k);},
                [this](uint32_t n,const std::vector<double>& v){pipelineDownOutput(n,v);},
                [this](bool ok){
                    if(!ok){finish(false);return;}accountWindow("down",downWindow_);
                    downNextK_=panels_[downConsumer_].firstTile+panels_[downConsumer_].tiles;
                    ++downPasses_;++pipelinePasses_;downActive_=false;
                },std::move(initial),[this](uint32_t k){return panelGM(downConsumer_)+uint64_t(k)*activeRows_*128;},false,{},
                [this]{return acquirePipelineArrays(2);},[this]{releasePipelineArrays(2);});
            return;
        }
    }
    void pipelineInitialDown(uint32_t n,ReuseWindowGemm::Loaded cb) {
        if(cycle_<downPartialReady_){downTask_=[this,n,cb]{pipelineInitialDown(n,cb);};return;}
        std::vector<uint8_t> raw;uint64_t ready=0;
        if(!wcp_->readProjectionPartial(uint64_t(n)*activeRows_*128,activeRows_*128,cycle_,raw,ready)){
            cb(false,{});return;
        }
        downCRead_+=raw.size();
        pipelineInitialReady(ready,std::move(raw),std::move(cb));
    }
    void pipelineInitialReady(uint64_t ready,std::vector<uint8_t> raw,ReuseWindowGemm::Loaded cb) {
        if(cycle_<ready){downTask_=[this,ready,raw,cb]{pipelineInitialReady(ready,raw,cb);};return;}
        ++downRestores_;cb(true,decode(raw));
    }
    void pipelineDownOutput(uint32_t n,const std::vector<double>& v) {
        const auto& p=panels_[downConsumer_];
        if(p.firstTile+p.tiles<d_.output_dim/64){
            auto raw=encode(v);uint64_t ready=0;
            if(!wcp_->writeProjectionPartial(uint64_t(n)*activeRows_*128,raw,cycle_,ready)){finish(false);return;}
            downCWrite_+=raw.size();downPartialReady_=std::max(downPartialReady_,ready);
            downWindow_.delayUntil(ready);downWindow_.resumeOutput();return;
        }
        // One output-tile snapshot, bounded to 8 KiB. Strided writes preserve
        // the existing row-major Down/residual epilogue and its ACK boundary.
        pipelineWrite_=encode(v);pipelineWriteN_=n;pipelineWriteOffset_=pipelineWriteNext_=0;
    }
    void pumpPipelineWrite() {
        if(pipelineWrite_.empty())return;
        if(!downOutputCount_){
            if(downPanels_[downOutputIndex_].draining || downPanels_[downOutputIndex_].epilogue)return;
            downOutputStart_=pipelineWriteN_;
        }
        const uint32_t tiles=std::min(16u,d_.input_dim/64-downOutputStart_);
        while(!pipelineWrite_.empty() && pipelineWritePending_<writeWindow_ && pipelineWriteNext_<pipelineWrite_.size()){
            const size_t offset=pipelineWriteNext_,row=offset/128,inRow=offset%128;
            const size_t len=std::min<size_t>(128-inRow,gm_->localMaxRequestBytes());
            const uint64_t addr=downOutputGM(downOutputIndex_)+(row*tiles+downOutputCount_)*128+inRow;
            std::vector<uint8_t> raw(pipelineWrite_.begin()+offset,pipelineWrite_.begin()+offset+len);
            ++pipelineWritePending_;pipelineWriteNext_+=len;
            if(!gm_->localWriteAsync(addr,raw,LocalMemoryClient::RoCC,++tag_,[this,len,tiles](bool ok,uint64_t){
                --pipelineWritePending_;if(!ok){finish(false);return;}
                localWrite_+=len;pipelineWriteOffset_+=len;
                if(pipelineWriteOffset_!=pipelineWrite_.size())return;
                pipelineWrite_.clear();
                if(++downOutputCount_==tiles){
                    auto& p=downPanels_[downOutputIndex_];p.row=row_;p.rows=activeRows_;
                    p.firstTile=downOutputStart_;p.tiles=tiles;p.nextRow=p.inFlight=0;
                    p.epilogue=d_.residual_offset!=0;p.draining=!p.epilogue;
                    downOutputCount_=0;downOutputIndex_=1-downOutputIndex_;
                }
                downWindow_.resumeOutput();
            })){--pipelineWritePending_;pipelineWriteNext_-=len;return;}
        }
    }
    ReuseWindowGemm::Fetch weightFetcher(){
        if(!sharedWeights_)return {};
        return [this](uint64_t source,uint64_t destination,std::function<void(bool)> done){
            sharing_->fetch(d_.epoch,source,destination,std::move(done));
        };
    }
    using Task = std::function<void()>;
    struct Panel {
        uint32_t row=0, rows=0, firstTile=0, tiles=0, nextRow=0, inFlight=0;
        bool draining=false,epilogue=false;
    };
    struct BackgroundRead {
        std::vector<uint8_t> data;
        uint64_t address=0;
        size_t offset=0;
        uint32_t tile=0,k=0;
        bool busy=false,pending=false,ready=false,up=false;
        uint8_t phase=0; // 0 idle, 1 HBM DMA, 2 GM read, 3 ready
        bool weight=false;
        uint64_t submitted=0,dmaDone=0,gmDone=0,source=0;
    };
    enum class FfnState { Idle, Reading, IssueSilu, WaitSilu, IssueMultiply, WaitMultiply, Writing };
    struct FfnChunk {
        FfnState state=FfnState::Idle;
        BackgroundRead gate,up;
        size_t offset=0,elements=0,writeOffset=0;
        bool writePending=false;
        std::vector<double> input,coefficients;
        std::vector<uint8_t> bytes;
        SFUVectorResult result{};
    };
    enum class ResidualState { Idle, Reading, Issue, Wait, Writing };
    static uint32_t option(const char* name,uint32_t fallback,uint32_t minimum,uint32_t maximum) {
        const char* value=std::getenv(name); if (!value) return fallback;
        char* end=nullptr; const unsigned long parsed=std::strtoul(value,&end,10);
        if (!*value || *end || parsed<minimum || parsed>maximum)
            throw std::invalid_argument(name);
        return static_cast<uint32_t>(parsed);
    }
    bool panelEnabled() const { return isFfn() || (d_.op==1 && !singleRow_ && panelLimit_); }
    uint32_t panelTiles() const { return isFfn() ? ffnPanelLimit_ : panelLimit_; }
    uint64_t panelGM(uint32_t index) const { return outGM()+index*DecoderMemoryLayout::decoder_panel_bytes; }
    uint64_t upPanelGM(uint32_t index) const { return gm_->getBaseAddr()+DecoderMemoryLayout::decoder_ffn_up_gm+index*DecoderMemoryLayout::decoder_panel_bytes; }
    uint64_t producerPanelGM() const { return isFfn() && ffnUp_ ? upPanelGM(panelIndex_) : panelGM(panelIndex_); }
    uint32_t core() const { return 4 + d_.manager + d_.lane * 4; }
    uint64_t inputGM() const { return gm_->getBaseAddr()+(useNormCache_?
        DecoderMemoryLayout::decoder_norm_handoff_gm:DecoderMemoryLayout::decoder_input_gm); }
    uint64_t weightGM() const { return gm_->getBaseAddr() + DecoderMemoryLayout::decoder_weight_gm; }
    uint64_t outGM() const { return gm_->getBaseAddr() + DecoderMemoryLayout::decoder_output_gm; }
    void defer(Task f) { next_ = std::move(f); }
    uint64_t chainGM() const {return gm_->getBaseAddr()+DecoderMemoryLayout::decoder_norm_handoff_gm;}
    uint64_t chainWoGM() const {return gm_->getBaseAddr()+DecoderMemoryLayout::decoder_wo_handoff_gm;}
    void finish(bool ok) {
        if(elideOutput_){
            const uint64_t expected=uint64_t(std::min(64u,d_.rows-d_.lane*64))*d_.input_dim*2;
            if(ok && (!layerPublished_ || layerBytes_!=expected || elidedOutputBytes_!=expected))ok=false;
            std::printf("[DECODER_OUTPUT_RESIDENCY] core=%u epoch=%u stage=%u local_produced=%llu saved_hbm_write=%llu published=%u capacity_added=0 status=%u\n",
                core(),d_.epoch,d_.stage,(unsigned long long)layerBytes_,
                (unsigned long long)elidedOutputBytes_,layerPublished_?1u:0u,ok?0u:1u);
        }
        if(lmInputResidency_ && singleRow_ && d_.stage==11){
            lmInputValid_=ok;lmInputDesc_=d_;
            std::printf("[LM_INPUT_RESIDENCY] core=%u epoch=%u hit=%u saved_hbm_read=%llu capacity_added=0 status=%u\n",
                core(),d_.epoch,useLmInput_?1u:0u,(unsigned long long)lmInputSaved_,ok?0u:1u);
        }
        if(layerProducer_ || layerConsumer_)std::printf("[SOFTWARE_LAYER_RESIDENCY] core=%u epoch=%u stage=%u producer=%u bytes=%llu saved_hbm_read=%llu metadata_bytes=64 capacity_added=0 status=%u\n",
            core(),d_.epoch,d_.stage,layerProducer_?1u:0u,(unsigned long long)layerBytes_,
            (unsigned long long)layerSaved_,ok?0u:1u);
        if(layerResidualConsumer_)std::printf("[DECODER_INPUT_RESIDENCY] core=%u epoch=%u stage=%u saved_hbm_read=%llu metadata_bytes=64 status=%u\n",
            core(),d_.epoch,d_.stage,(unsigned long long)layerResidualSaved_,ok?0u:1u);
        if(chainJob_) {
            const uint64_t rowBytes=uint64_t(std::min(64u,d_.rows-d_.lane*64))*d_.input_dim*2;
            if(ok && chainJob_<4 && chainLocalBytes_!=rowBytes*(fusedAddNorm_?2:1))ok=false;
            if(ok && chainJob_==1)chainState_=1;
            else if(ok && chainJob_==2){
                chainState_=fusedAddNorm_?3:2;chainH1Address_=d_.output_addr;
                if(fusedAddNorm_)chainNormAddress_=(d_.manager+1)*d_.node_stride+d_.head_dim;
            }
            else if(ok && chainJob_==3){chainState_=3;chainNormAddress_=d_.output_addr;}
            else chainState_=0;
            std::printf("[DECODER_STAGE_CHAIN] core=%u epoch=%u stage=%u step=%u state=%u row_bytes=%llu local_produced=%llu saved_hbm_read=%llu saved_hbm_write=%llu capacity_added=0 status=%u\n",
                core(),d_.epoch,d_.stage,chainJob_,chainState_,(unsigned long long)rowBytes,
                (unsigned long long)chainLocalBytes_,(unsigned long long)chainReadSaved_,
                (unsigned long long)chainWriteSaved_,ok?0u:1u);
        }
        if(fusedAddNorm_)std::printf("[DECODER_ADD_NORM_FUSION] core=%u epoch=%u stage=%u chunks=%llu saved_gm_read=%llu norm_bytes=%llu fp16_boundary=1 capacity_added=0 status=%u\n",
            core(),d_.epoch,d_.stage,(unsigned long long)fusedNormChunks_,
            (unsigned long long)fusedNormReadSaved_,(unsigned long long)normCacheWritten_,ok?0u:1u);
        if(cacheNorm_)normCacheValid_=ok && normCacheWritten_==uint64_t(std::min(64u,d_.rows-d_.lane*64))*d_.input_dim*2;
        if (downOwnedBuffer_) { wcp_->endProjectionPartialBuffer(); downOwnedBuffer_=false; }
        status_ = ok ? 0 : 1; active_ = false; done_ = true; next_ = {};
        if(cacheNorm_ || useNormCache_)std::printf("[DECODER_NORM_HANDOFF] core=%u epoch=%u stage=%u producer=%u consumer=%u cache_bytes=%llu saved_hbm_read=%llu materialized=1 capacity_added=0\n",
            core(),d_.epoch,d_.stage,cacheNorm_?1u:0u,useNormCache_?1u:0u,
            (unsigned long long)normCacheWritten_,(unsigned long long)normCacheReadSaved_);
        if(allGemmWindow_ && lmGrouped()) std::printf("[GEMM_REUSE_WINDOW] core=%u stage=%u epoch=%u operator=lm_head row=0 reuse_n=%u reuse_m=1 k_window=%u windows=%llu launches=%llu hbm_read=%llu gm_read=%llu gm_write=%llu c_drains=%llu c_restores=0 status=%u\n",
            core(),d_.stage,d_.epoch,lmParallel_,d_.input_dim/64,(unsigned long long)lmLaunches_,
            (unsigned long long)lmArrayTiles_,(unsigned long long)hbmRead_,(unsigned long long)localRead_,
            (unsigned long long)localWrite_,(unsigned long long)(d_.input_dim?lmLaunches_/(d_.input_dim/64):0),ok?0:1);
        if (d_.op==1 || isFfn()) std::printf("[DECODER_DATAFLOW] core=%u stage=%u epoch=%u panel_tiles=%u output_requests=%llu output_high_water=%llu prefetch=%u prefetch_requests=%llu prefetch_hits=%llu prefetch_wait_cycles=%llu\n",
            core(),d_.stage,d_.epoch,panelEnabled()?panelTiles():0,
            (unsigned long long)outputRequests_,(unsigned long long)outputHighWater_,prefetchEnabled_,
            (unsigned long long)prefetchRequests_,(unsigned long long)prefetchHits_,(unsigned long long)prefetchWaitCycles_);
        if (singleRow_) std::printf("[DECODER_LM_FLOW] core=%u stage=%u epoch=%u n_parallel=%u launches=%llu array_k_tiles=%llu\n",
            core(),d_.stage,d_.epoch,lmParallel_,(unsigned long long)lmLaunches_,(unsigned long long)lmArrayTiles_);
        if (isFfn()) std::printf("[DECODER_FFN_FLOW] core=%u stage=%u epoch=%u panels_produced=%llu panels_consumed=%llu vector_chunks=%llu gemm_sfu_overlap_cycles=%llu slot_wait_cycles=%llu gm_required_bytes=%llu\n",
            core(),d_.stage,d_.epoch,(unsigned long long)ffnProduced_,(unsigned long long)ffnConsumed_,
            (unsigned long long)ffnChunks_,(unsigned long long)ffnOverlapCycles_,(unsigned long long)ffnSlotWaitCycles_,
            (unsigned long long)(d_.op==9?DecoderMemoryLayout::decoder_down_required_gm:DecoderMemoryLayout::decoder_required_gm));
        if(isFfn())std::printf("[DECODER_SFU_STREAM] core=%u stage=%u epoch=%u enabled=%u contexts=2 chunk_elements=%u high_water=%llu io_compute_overlap=%llu drained=%u\n",
            core(),d_.stage,d_.epoch,streamFfn_?1u:0u,ffnChunkLimit_,
            (unsigned long long)ffnStreamHighWater_,(unsigned long long)ffnStreamOverlap_,
            (ffnStream_[0].state==FfnState::Idle && ffnStream_[1].state==FfnState::Idle)?1u:0u);
        if (isFfn()) std::printf("[DECODER_PREFETCH_FLOW] core=%u stage=%u epoch=%u enabled=%u slots=%u requests=%llu hits=%llu wait_total=%llu wait_dma=%llu wait_gm=%llu wait_program=%llu wait_array=%llu dma_sum=%llu gm_sum=%llu program_sum=%llu compute_sum=%llu\n",
            core(),d_.stage,d_.epoch,prefetchEnabled_?1u:0u,prefetchSlots_,
            (unsigned long long)prefetchRequests_,(unsigned long long)prefetchHits_,
            (unsigned long long)prefetchWaitCycles_,(unsigned long long)prefetchDmaWaitCycles_,
            (unsigned long long)prefetchGmWaitCycles_,(unsigned long long)prefetchProgramWaitCycles_,
            (unsigned long long)arrayWaitCycles_,(unsigned long long)weightDmaCycles_,
            (unsigned long long)weightGmCycles_,(unsigned long long)weightProgramCycles_,
            (unsigned long long)arrayComputeCycles_);
        if (isFfn()) std::printf("[DECODER_GROUP_FLOW] core=%u stage=%u epoch=%u groups=%u launches=%llu program_cycles=%llu compute_array_cycles=%llu both_groups_compute_cycles=%llu\n",
            core(),d_.stage,d_.epoch,arrayGroups_,(unsigned long long)groupLaunches_,
            (unsigned long long)groupProgramCycles_,(unsigned long long)groupComputeArrayCycles_,
            (unsigned long long)groupOverlapCycles_);
        if(pipelined())std::printf("[DECODER_BLOCK_PIPELINE] core=%u epoch=%u stage=%u panels=%llu both_active=%llu producer_transfer_down_compute=%llu down_transfer_producer_compute=%llu lease_wait=%llu arrays=64 contexts=1 extra_gm_capacity=0 policy=%u owner_at_finish=%u pending_at_finish=%u\n",
            core(),d_.epoch,d_.stage,(unsigned long long)pipelinePasses_,(unsigned long long)pipelineBothActive_,
            (unsigned long long)pipelineProducerTransfers_,(unsigned long long)pipelineDownTransfers_,
            (unsigned long long)pipelineLeaseWait_,pipelinePolicy_,arrayOwner_,pipelineWritePending_);
        if (d_.op==9) std::printf("[DECODER_GM_FLOW] core=%u stage=%u epoch=%u layout=%u down_input_reuse=%u\n",core(),d_.stage,d_.epoch,ffnLayout_,downInputReuse_?1u:0u);
        if (d_.op==9) std::printf("[DECODER_DOWN_FLOW] core=%u stage=%u epoch=%u panels_consumed=%llu cbuffer_bytes=%llu cbuffer_read_bytes=%llu cbuffer_write_bytes=%llu restores=%llu\n",
            core(),d_.stage,d_.epoch,(unsigned long long)downPasses_,(unsigned long long)(64ull*d_.input_dim*2),
            (unsigned long long)downCRead_,(unsigned long long)downCWrite_,(unsigned long long)downRestores_);
        if (d_.op==9 && d_.residual_offset) std::printf("[DECODER_RESIDUAL_FLOW] core=%u stage=%u epoch=%u chunks=%llu residual_hbm_bytes=%llu gemm_overlap_cycles=%llu scratch_bytes=8192\n",
            core(),d_.stage,d_.epoch,(unsigned long long)residualChunks_,
            (unsigned long long)residualBytes_,(unsigned long long)residualOverlap_);
        std::printf("[DECODER_OP] event=done core=%u stage=%u op=%u tick=%llu start_tick=%llu status=%llu hbm_read=%llu hbm_write=%llu gm_read=%llu gm_write=%llu epoch=%u\n",
            core(), d_.stage, d_.op, (unsigned long long)tick_, (unsigned long long)startTick_,
            (unsigned long long)status_, (unsigned long long)hbmRead_, (unsigned long long)hbmWrite_,
            (unsigned long long)localRead_, (unsigned long long)localWrite_, d_.epoch);
    }
    static std::vector<double> decode(const std::vector<uint8_t>& b) {
        std::vector<double> v(b.size()/2);
        for (size_t i=0; i<v.size(); ++i) { uint16_t x; std::memcpy(&x,b.data()+2*i,2); v[i]=golem_fp16_to_float(x); }
        return v;
    }
    static std::vector<uint8_t> encode(const std::vector<double>& v) {
        std::vector<uint8_t> b(v.size()*2);
        for (size_t i=0; i<v.size(); ++i) { const uint16_t x=golem_float_to_fp16(v[i]); std::memcpy(b.data()+2*i,&x,2); }
        return b;
    }
    // Every tensor snapshot is read/written through the timed local port.
    void readLocal(uint64_t addr, size_t size, std::function<void(const std::vector<uint8_t>&)> cb) {
        localBuffer_.assign(size,0); localOffset_=0; localAddr_=addr; readDone_=std::move(cb);
        defer([this] { readChunk(); });
    }
    void readChunk() {
        const size_t len=std::min(gm_->localMaxRequestBytes(),localBuffer_.size()-localOffset_);
        if (!gm_->localReadAsync(localAddr_+localOffset_,len,LocalMemoryClient::RoCC,++tag_,
            [this,len](bool ok,uint64_t,const std::vector<uint8_t>& b) {
                if (!ok || b.size()!=len) { finish(false); return; }
                std::copy(b.begin(),b.end(),localBuffer_.begin()+localOffset_);
                localOffset_+=len; localRead_+=len;
                if (localOffset_==localBuffer_.size()) {
                    auto f=std::move(readDone_); auto b=std::move(localBuffer_); f(b);
                } else defer([this] { readChunk(); });
            })) defer([this] { readChunk(); });
    }
    void writeLocal(uint64_t addr,const std::vector<uint8_t>& b,Task cb) {
        localBuffer_=b; localOffset_=0; localAddr_=addr; writeDone_=std::move(cb);
        defer([this] { writeChunk(); });
    }
    void writeChunk() {
        const size_t len=std::min(gm_->localMaxRequestBytes(),localBuffer_.size()-localOffset_);
        std::vector<uint8_t> b(localBuffer_.begin()+localOffset_,localBuffer_.begin()+localOffset_+len);
        if (!gm_->localWriteAsync(localAddr_+localOffset_,b,LocalMemoryClient::RoCC,++tag_,
            [this,len](bool ok,uint64_t) {
                if (!ok) { finish(false); return; } localOffset_+=len; localWrite_+=len;
                if (localOffset_==localBuffer_.size()) { localBuffer_.clear(); auto f=std::move(writeDone_); f(); }
                else defer([this] { writeChunk(); });
            })) defer([this] { writeChunk(); });
    }
    // Scatter one existing gathered tile into the final row-major GM panel.
    // No new tensor buffer: localBuffer_ owns the same <=8 KiB output snapshot.
    // At most writeWindow_ requests use the existing GM pending queue.
    void writePanelTile(uint64_t address,uint32_t tiles,uint32_t tile,
                        const std::vector<uint8_t>& bytes,Task done) {
        if (tiles==1) { writeLocal(address,bytes,std::move(done));return; }
        localBuffer_=bytes;localAddr_=address+uint64_t(tile)*128;
        stridedStride_=uint64_t(tiles)*128;stridedNext_=stridedDone_=stridedPending_=0;
        writeDone_=std::move(done);defer([this] { pumpPanelTile(); });
    }
    void pumpPanelTile() {
        const size_t rows=localBuffer_.size()/128;
        if (stridedDone_==rows) {
            localBuffer_.clear();auto done=std::move(writeDone_);done();return;
        }
        if (stridedNext_<rows && stridedPending_<writeWindow_) {
            const size_t row=stridedNext_;
            std::vector<uint8_t> bytes(localBuffer_.begin()+row*128,localBuffer_.begin()+(row+1)*128);
            ++stridedNext_;++stridedPending_;
            if (!gm_->localWriteAsync(localAddr_+row*stridedStride_,bytes,LocalMemoryClient::RoCC,++tag_,
                [this](bool ok,uint64_t) {
                    --stridedPending_;
                    if (!ok) { finish(false);return; }
                    ++stridedDone_;localWrite_+=128;
                })) { --stridedNext_;--stridedPending_; }
        }
        defer([this] { pumpPanelTile(); });
    }
    void loadHBM(uint64_t src,uint64_t dst,size_t bytes,Task cb) {
        hbmRead_+=bytes;
        gm_->dma_read_from_host_to_globalmem(src,bytes,dst,[this,cb](bool ok) {
            if (!ok) finish(false); else defer(cb);
        });
    }
    void storeHBM(uint64_t src,uint64_t dst,size_t bytes,Task cb) {
        hbmWrite_+=bytes;
        gm_->dma_write_from_globalmem_to_host(src,dst,bytes,[this,cb](bool ok) {
            if (!ok) finish(false); else defer(cb);
        });
    }
    void loadRows() {
        if (row_>=d_.rows) { signal(); return; }
        if (d_.op==9) downNextK_=0;
        activeRows_=std::min(64u,d_.rows-row_);
        arrays_.resize(activeRows_);
        for (uint32_t i=0;i<activeRows_;++i) arrays_[i]=i;
        if (d_.op==6) { loadEmbeddingTokens(); return; }
        if (d_.op==1 || isFfn()) {
            ffnUp_=false;
            panelCount_=0;
            nt_=singleRow_ ? d_.manager+d_.lane*4 : 0; kt_=0;
            if (nt_>=d_.output_dim/64) { signal(); return; }
            if(useLmInput_){lmInputSaved_+=d_.input_dim*2;defer([this]{loadWeight();});return;}
            if(chainJob_==4){chainReadSaved_+=uint64_t(activeRows_)*d_.input_dim*2;defer([this]{loadWeight();});return;}
            if(useNormCache_){normCacheReadSaved_+=uint64_t(activeRows_)*d_.input_dim*2;defer([this]{loadWeight();});return;}
            if (d_.op==1 && d_.head_dim) { head_=0; gathered_.assign(activeRows_*d_.input_dim*2,0); gatherHead(); }
            else loadHBM(d_.input_addr+uint64_t(row_)*d_.input_dim*2,inputGM(),activeRows_*d_.input_dim*2,
                [this] { packInput(); });
        } else {
            // Four rows (at most 8192 elements) fit bounded vector scratch.
            vectorRow_=row_; vectorRows_=std::min(4u,64u);
            loadVector();
        }
    }
    void gatherHead() {
        const size_t bytes=activeRows_*d_.head_dim*2;
        loadHBM(d_.input_addr+uint64_t(head_)*d_.rows*d_.head_dim*2+uint64_t(row_)*d_.head_dim*2,
            weightGM(),bytes,[this,bytes] {
                readLocal(weightGM(),bytes,[this](const std::vector<uint8_t>& b) {
                    for (uint32_t r=0;r<activeRows_;++r) std::copy(b.begin()+r*d_.head_dim*2,b.begin()+(r+1)*d_.head_dim*2,
                        gathered_.begin()+(uint64_t(r)*d_.input_dim+head_*d_.head_dim)*2);
                    if (++head_==d_.input_dim/d_.head_dim) {
                        auto b=std::move(gathered_); writeLocal(inputGM(),b,[this] { packInput(); });
                    } else defer([this] { gatherHead(); });
                });
            });
    }
    void loadEmbeddingTokens() {
        loadHBM(d_.input_addr+uint64_t(row_)*4,weightGM(),activeRows_*4,[this] {
            readLocal(weightGM(),activeRows_*4,[this](const std::vector<uint8_t>& bytes) {
                embeddingTokens_.resize(activeRows_);
                for (uint32_t r=0;r<activeRows_;++r) {
                    std::memcpy(&embeddingTokens_[r],bytes.data()+r*4,4);
                    if (embeddingTokens_[r]>=d_.input_dim) { finish(false); return; }
                }
                embeddingRow_=0; loadEmbeddingRow();
            });
        });
    }
    void loadEmbeddingRow() {
        const uint32_t token=embeddingTokens_[embeddingRow_];
        const uint64_t address=(1+uint64_t(token%4))*d_.node_stride+d_.weights_offset+uint64_t(token/4)*d_.output_dim*2;
        loadHBM(address,inputGM(),d_.output_dim*2,[this] {
            storeHBM(inputGM(),d_.output_addr+uint64_t(row_+embeddingRow_)*d_.output_dim*2,d_.output_dim*2,[this] {
                if (++embeddingRow_<activeRows_) loadEmbeddingRow();
                else { embeddingTokens_.clear(); row_+=256; loadRows(); }
            });
        });
    }
    void packInput() {
        readLocal(inputGM(),activeRows_*d_.input_dim*2,[this](const std::vector<uint8_t>& b) {
            std::vector<uint8_t> packed(b.size());
            for (uint32_t k=0;k<d_.input_dim/64;++k) for (uint32_t r=0;r<activeRows_;++r)
                std::copy(b.begin()+(uint64_t(r)*d_.input_dim+k*64)*2,
                    b.begin()+(uint64_t(r)*d_.input_dim+k*64+64)*2,packed.begin()+(uint64_t(k)*activeRows_*64+r*64)*2);
            writeLocal(inputGM(),packed,[this] { loadWeight(); });
        });
    }
    bool lmGrouped() const { return singleRow_ && lmParallel_>1; }
    void loadWeight() {
        if (lmGrouped()) { lmBegin(); return; }
        if (d_.op==9 && !pipelined() && !ffnUp_ && !panelCount_ && !kt_ &&
            startDown([this] { loadWeight(); })) return;
        if (panelEnabled() && !panelCount_) {
            const bool consuming=isFfn() && ffnState_!=FfnState::Idle && ffnConsumer_==panelIndex_;
            if (panels_[panelIndex_].draining || ffnReady_[panelIndex_] || downReady_[panelIndex_] || consuming ||
                (downActive_ && downConsumer_==panelIndex_)) {
                if(pipelined() && pipelinePolicy_==2){
                    // Output is still owned by the consumer. Prepare only A/B;
                    // the array lease prevents computation until the slot frees.
                    panelStart_=nt_;startProducerWindow();return;
                }
                if (isFfn()) ++ffnSlotWaitCycles_;
                defer([this] { loadWeight(); }); return;
            }
            panelStart_=nt_;
        }
        if (allGemmWindow_) { startProducerWindow(); return; }
        if (prefetchEnabled_) {
            auto& read=weightReads_[weightSlot_];
            if (!read.busy && !read.ready && !read.pending) beginWeightRead(weightSlot_,nt_,kt_,ffnUp_);
            if (!read.ready) {
                ++prefetchWaitCycles_;
                if (read.phase==1) ++prefetchDmaWaitCycles_;
                else if (read.phase==2 || read.busy) ++prefetchGmWaitCycles_;
                else ++prefetchProgramWaitCycles_;
                defer([this] { loadWeight(); }); return;
            }
            if (read.tile!=nt_ || read.k!=kt_ || read.up!=ffnUp_) { finish(false); return; }
            traceWeight(read,false);operand_=decode(read.data); read=BackgroundRead{}; ++prefetchHits_;
            weightSlot_=(weightSlot_+1)%prefetchSlots_;
            scheduleWeightPrefetch(nt_,kt_,ffnUp_);
            inputRead_=BackgroundRead{};
            inputRead_.address=inputGM()+uint64_t(kt_)*activeRows_*128;
            inputRead_.data.resize(activeRows_*128); inputRead_.busy=true;
            defer([this] { programWeight(); });
            return;
        }
        const uint64_t addr=weightAddress(nt_,kt_,ffnUp_);
        loadHBM(addr,weightGM(),8192,[this] {
            readLocal(weightGM(),8192,[this](const std::vector<uint8_t>& b) {
                operand_=decode(b); defer([this] { programWeight(); });
            });
        });
    }
    uint64_t weightAddress(uint32_t n,uint32_t k,bool up=false) const {
        const uint64_t tile=(d_.op==1?uint64_t(d_.output_tile_begin):0)+n;
        return (1+tile%4)*d_.node_stride+(up?d_.input1_addr:d_.weights_offset)+((tile/4)*(d_.input_dim/64)+k)*8192;
    }
    bool nextWeight(uint32_t n,uint32_t k,bool up,uint32_t& nextN,uint32_t& nextK,bool& nextUp) const {
        if (lmGrouped()) {
            const uint32_t owner=d_.manager+d_.lane*4;
            const uint32_t begin=owner+((n-owner)/(16*lmParallel_))*(16*lmParallel_);
            nextN=n+16; nextK=k; nextUp=false;
            if (nextN<std::min(begin+16*lmParallel_,d_.output_dim/64)) return true;
            nextN=begin; nextK=k+1;
            if (nextK<d_.input_dim/64) return true;
            nextN=begin+16*lmParallel_;nextK=0;
            return nextN<d_.output_dim/64;
        }
        nextK=k+1; nextN=n; nextUp=up;
        if (nextK<d_.input_dim/64) return true;
        nextN=n+(singleRow_?16:1); nextK=0;
        if (isFfn()) {
            const uint32_t begin=(n/ffnPanelLimit_)*ffnPanelLimit_;
            const uint32_t end=std::min(begin+ffnPanelLimit_,d_.output_dim/64);
            if (nextN==end) {
                if (!up) { nextN=begin;nextUp=true;return true; }
                nextUp=false;
            }
        }
        return nextN<d_.output_dim/64;
    }
    bool hasWeightRead(uint32_t n,uint32_t k,bool up) const {
        for (uint32_t i=0;i<prefetchSlots_;++i) {
            const auto& r=weightReads_[i];
            if ((r.pending||r.busy||r.ready) && r.tile==n && r.k==k && r.up==up) return true;
        }
        return false;
    }
    void scheduleWeightPrefetch(uint32_t n,uint32_t k,bool up) {
        uint32_t slot=weightSlot_,nextN=n,nextK=k; bool nextUp=up;
        for (uint32_t i=0;i+1<prefetchSlots_;++i) {
            if (!nextWeight(nextN,nextK,nextUp,nextN,nextK,nextUp)) return;
            if (hasWeightRead(nextN,nextK,nextUp)) { slot=(slot+1)%prefetchSlots_;continue; }
            auto& r=weightReads_[slot];
            if (r.pending||r.busy||r.ready) return;
            beginWeightRead(slot,nextN,nextK,nextUp);
            slot=(slot+1)%prefetchSlots_;
        }
    }
    void beginWeightRead(uint32_t slot,uint32_t n,uint32_t k,bool up=false) {
        auto& read=weightReads_[slot];
        if (read.busy || read.ready || read.pending) { finish(false); return; }
        read.tile=n; read.k=k; read.up=up; read.pending=true; read.phase=1;
        read.weight=true;read.submitted=cycle_;read.source=weightAddress(n,k,up);
        read.address=gm_->getBaseAddr()+DecoderMemoryLayout::decoder_prefetch_gm+slot*8192;
        read.data.resize(8192); hbmRead_+=8192; ++prefetchRequests_;
        gm_->dma_read_from_host_to_globalmem(weightAddress(n,k,up),8192,read.address,[this,slot](bool ok) {
            auto& r=weightReads_[slot]; r.pending=false; r.phase=2;r.dmaDone=cycle_;
            if (!ok) { finish(false); return; } r.busy=true;
        });
    }
    void pumpRead(BackgroundRead& read) {
        if (!read.busy || read.pending || !active_) return;
        const size_t len=std::min(gm_->localMaxRequestBytes(),read.data.size()-read.offset);
        const uint64_t tag=++tag_; read.pending=true;
        if (!gm_->localReadAsync(read.address+read.offset,len,LocalMemoryClient::RoCC,tag,
            [this,&read,len](bool ok,uint64_t,const std::vector<uint8_t>& b) {
                read.pending=false;
                if (!ok || b.size()!=len) { finish(false); return; }
                std::copy(b.begin(),b.end(),read.data.begin()+read.offset);
                read.offset+=len; localRead_+=len;
                if (read.offset==read.data.size()) { read.busy=false; read.ready=true; read.phase=3;read.gmDone=cycle_; }
            })) read.pending=false;
    }
    void traceWeight(const BackgroundRead& r,bool down) {
        weightDmaCycles_+=r.dmaDone-r.submitted;
        weightGmCycles_+=r.gmDone-r.dmaDone;
        if (weightTrace_) std::printf("[DECODER_WEIGHT_TILE] core=%u stage=%u epoch=%u row=%u path=%u n=%u k=%u address=%llu submit=%llu dma_done=%llu gm_done=%llu consume=%llu\n",
            core(),d_.stage,d_.epoch,row_,down?2u:(r.up?1u:0u),r.tile,r.k,
            (unsigned long long)r.source,(unsigned long long)r.submitted,(unsigned long long)r.dmaDone,
            (unsigned long long)r.gmDone,(unsigned long long)cycle_);
    }
    void programWeight() {
        if (grouped()) { beginGroupedTile(false); return; }
        const uint64_t submitted=cycle_;
        if (!wcp_->programGemmMatrixGroupClassBankAsync(arrays_,0,operand_,2,
            AttentionClusterTrafficClass::ProjectionWeights,++tag_,cycle_,[this,submitted](bool ok,uint64_t) {
                weightProgramCycles_+=cycle_-submitted;
                if (!ok) { finish(false); return; } operand_.clear();
                if (prefetchEnabled_) { defer([this] { consumeInputRead(); }); }
                else readLocal(inputGM()+uint64_t(kt_)*activeRows_*128,activeRows_*128,[this](const std::vector<uint8_t>& b) {
                    operand_=decode(b); defer([this] { programInput(); });
                });
            })) { ++prefetchProgramWaitCycles_; defer([this] { programWeight(); }); }
    }
    void consumeInputRead() {
        if (!inputRead_.ready) { defer([this] { consumeInputRead(); }); return; }
        operand_=decode(inputRead_.data); inputRead_=BackgroundRead{};
        defer([this] { programInput(); });
    }
    void programInput() {
        if (!wcp_->programGemmInputScatterBankAsync(arrays_,0,operand_,2,
            AttentionClusterTrafficClass::ProjectionInputScatter,++tag_,cycle_,[this](bool ok,uint64_t) {
                if (!ok) { finish(false); return; } operand_.clear(); defer([this] { launch(); });
            })) defer([this] { programInput(); });
    }
    void launch() {
        const uint64_t launched=cycle_;
        remaining_=activeRows_;
        if (!wcp_->launchGemmArrayGroupActiveBank(arrays_,0,kt_?1:0,64,cycle_,[this,launched](uint32_t,uint64_t) {
            if (--remaining_) return;
            arrayComputeCycles_+=cycle_-launched;
            if (++kt_<d_.input_dim/64) defer([this] { loadWeight(); });
            else defer([this] { readOutput(); });
            })) { ++arrayWaitCycles_; defer([this] { launch(); }); }
    }
    // N parallelism reuses existing single-context arrays. Each distinct
    // matrix crosses the timed shared port separately; only the input is shared.
    // Array i owns nt_+16*i. Every output retains the original K order.
    void lmBegin() {
        lmWidth_=std::min(lmParallel_,(d_.output_dim/64-1-nt_)/16+1);
        arrays_.resize(lmWidth_);
        for (uint32_t i=0;i<lmWidth_;++i) arrays_[i]=i;
        lmIndex_=0; lmLoadWeight();
    }
    void lmLoadWeight() {
        const uint32_t n=nt_+16*lmIndex_;
        if (prefetchEnabled_) {
            auto& r=weightReads_[weightSlot_];
            if (!r.busy && !r.ready && !r.pending) beginWeightRead(weightSlot_,n,kt_);
            if (!r.ready) { ++prefetchWaitCycles_;defer([this] { lmLoadWeight(); });return; }
            if (r.tile!=n || r.k!=kt_ || r.up) { finish(false);return; }
            traceWeight(r,false);operand_=decode(r.data);r=BackgroundRead{};++prefetchHits_;
            weightSlot_=(weightSlot_+1)%prefetchSlots_;
            scheduleWeightPrefetch(n,kt_,false);
            defer([this] { lmProgramWeight(); });
        } else loadHBM(weightAddress(n,kt_),weightGM(),8192,[this] {
            readLocal(weightGM(),8192,[this](const std::vector<uint8_t>& b) {
                operand_=decode(b);defer([this] { lmProgramWeight(); });
            });
        });
    }
    void lmProgramWeight() {
        const uint64_t submitted=cycle_;
        if (!wcp_->programGemmMatrixGroupClassBankAsync({lmIndex_},0,operand_,2,
            AttentionClusterTrafficClass::ProjectionWeights,++tag_,cycle_,[this,submitted](bool ok,uint64_t) {
                weightProgramCycles_+=cycle_-submitted;
                if (!ok) { finish(false);return; }operand_.clear();
                if (++lmIndex_<lmWidth_) defer([this] { lmLoadWeight(); });
                else readLocal(inputGM()+uint64_t(kt_)*128,128,[this](const std::vector<uint8_t>& b) {
                    const auto v=decode(b);operand_.clear();
                    for (uint32_t i=0;i<lmWidth_;++i) operand_.insert(operand_.end(),v.begin(),v.end());
                    defer([this] { lmScatter(); });
                });
            })) defer([this] { lmProgramWeight(); });
    }
    void lmScatter() {
        if (!wcp_->programGemmInputScatterBankAsync(arrays_,0,operand_,2,
            AttentionClusterTrafficClass::ProjectionInputScatter,++tag_,cycle_,[this](bool ok,uint64_t) {
                if (!ok) { finish(false);return; }operand_.clear();defer([this] { lmLaunch(); });
            })) defer([this] { lmScatter(); });
    }
    void lmLaunch() {
        const uint64_t launched=cycle_;remaining_=lmWidth_;
        if (!wcp_->launchGemmArrayGroupActiveBank(arrays_,0,kt_?1:0,64,cycle_,[this,launched](uint32_t,uint64_t) {
                if (--remaining_) return;
                arrayComputeCycles_+=cycle_-launched;
                if (++kt_<d_.input_dim/64) defer([this] { lmBegin(); });
                else defer([this] { lmOutput(); });
            })) defer([this] { lmLaunch(); });
        else { ++lmLaunches_;lmArrayTiles_+=lmWidth_; }
    }
    void lmOutput() {
        if (!wcp_->readGemmOutputGroupClassAsync(arrays_,2,AttentionClusterTrafficClass::ProjectionOutput,
            ++tag_,cycle_,[this](bool ok,uint64_t,const std::vector<double>& v) {
                if (!ok || v.size()!=lmWidth_*64) { finish(false);return; }
                writeLocal(outGM(),encode(v),[this] { lmIndex_=0;lmStore(); });
            })) defer([this] { lmOutput(); });
    }
    void lmStore() {
        ++outputRequests_;outputHighWater_=std::max<uint64_t>(outputHighWater_,1);
        storeHBM(outGM()+uint64_t(lmIndex_)*128,d_.output_addr+uint64_t(nt_+16*lmIndex_)*128,128,[this] {
            if (++lmIndex_<lmWidth_) { lmStore();return; }
            nt_+=16*lmWidth_;kt_=0;
            if (nt_<d_.output_dim/64) lmBegin();else { row_+=256;loadRows(); }
        });
    }
    void readOutput() {
        if (groupRemaining_[0] || groupRemaining_[1]) { defer([this] { readOutput(); });return; }
        if (!wcp_->readGemmOutputGroupClassAsync(arrays_,2,AttentionClusterTrafficClass::ProjectionOutput,
            ++tag_,cycle_,[this](bool ok,uint64_t,const std::vector<double>& v) {
                if (!ok || v.size()!=activeRows_*64) { finish(false); return; }
                if (panelEnabled()) {
                    const uint64_t address=producerPanelGM()+uint64_t(panelCount_)*activeRows_*128;
                    writeLocal(address,encode(v),[this] {
                        ++panelCount_; ++nt_;
                        if (panelCount_==panelTiles() || nt_==d_.output_dim/64) preparePanel();
                        else { kt_=0; loadWeight(); }
                    });
                } else writeLocal(outGM(),encode(v),[this] { storeRow_=0; storeTileRow(); });
            })) defer([this] { readOutput(); });
    }
    void storeTileRow() {
        ++outputRequests_;
        outputHighWater_=std::max<uint64_t>(outputHighWater_,1);
        storeHBM(outGM()+storeRow_*128,d_.output_addr+(uint64_t(row_+storeRow_)*d_.output_dim+nt_*64)*2,
            128,[this] {
                if (++storeRow_<activeRows_) { storeTileRow(); return; }
                nt_+=singleRow_ ? 16 : 1;
                if (nt_<d_.output_dim/64) { kt_=0; loadWeight(); }
                else { row_+=256; loadRows(); }
            });
    }
    void preparePanel() {
        // Array outputs are tile-major; HBM is row-major. Read and rewrite
        // through the modeled GM ports before issuing contiguous row panels.
        const uint32_t index=panelIndex_, tiles=panelCount_, rows=activeRows_;
        const uint64_t address=producerPanelGM();
        const size_t bytes=uint64_t(rows)*tiles*128;
        if (tileMajorFfn()) { panelPrepared(index,tiles,rows);return; }
        readLocal(address,bytes,[this,index,tiles,rows,address](const std::vector<uint8_t>& b) {
            std::vector<uint8_t> packed(b.size());
            for (uint32_t tile=0;tile<tiles;++tile) for (uint32_t row=0;row<rows;++row)
                std::copy_n(b.begin()+(uint64_t(tile)*rows+row)*128,128,
                    packed.begin()+(uint64_t(row)*tiles+tile)*128);
            writeLocal(address,packed,[this,index,tiles,rows] { panelPrepared(index,tiles,rows); });
        });
    }
    void panelPrepared(uint32_t index,uint32_t tiles,uint32_t rows) {
                auto& p=panels_[index];
                p.row=row_; p.rows=rows; p.firstTile=panelStart_; p.tiles=tiles;
                p.nextRow=p.inFlight=0;
                if (isFfn() && !ffnUp_) {
                    ffnUp_=true; nt_=panelStart_; kt_=0; panelCount_=0;
                    defer([this] { loadWeight(); }); return;
                }
                if (isFfn()) { ffnReady_[index]=true; ffnUp_=false; ++ffnProduced_;
                    if(pipelined())pipelineProducerBlock_=false; }
                else p.draining=true;
                panelCount_=0; panelIndex_=1-index;
                if (nt_<d_.output_dim/64) { kt_=0; defer([this] { loadWeight(); }); }
                else if (d_.op==9) defer([this] { waitDownRow(); });
                else { row_+=256; defer([this] { loadRows(); }); }
    }

    void pumpOutputs() {
        auto* outputs=d_.op==9?downPanels_:panels_;
        for (uint32_t index=0;index<2;++index) {
            auto& p=outputs[index];
            if (elideOutput_) {
                // The residual epilogue sets draining only after every timed
                // GM write ACK. Keep the payload for the next layer consumer.
                if (p.draining) {
                    if(p.inFlight || p.nextRow){finish(false);return;}
                    elidedOutputBytes_+=uint64_t(p.rows)*p.tiles*128;
                    p.nextRow=p.rows;p.draining=false;
                }
                continue;
            }
            while (active_ && p.draining && p.nextRow<p.rows && totalOutputInFlight()<writeWindow_) {
                const uint32_t row=p.nextRow++; ++p.inFlight;
                const size_t bytes=p.tiles*128;
                ++outputRequests_; hbmWrite_+=bytes;
                outputHighWater_=std::max<uint64_t>(outputHighWater_,totalOutputInFlight());
                const uint32_t width=d_.op==9?d_.input_dim:d_.output_dim;
                const uint64_t source=layerProducer_?chainGM()+
                    ((uint64_t(p.row-d_.lane*64)+row)*width+p.firstTile*64)*2:
                    (d_.op==9?downOutputGM(index):panelGM(index))+uint64_t(row)*bytes;
                gm_->dma_write_from_globalmem_to_host(source,
                    d_.output_addr+(uint64_t(p.row+row)*width+p.firstTile*64)*2,bytes,
                    [this,index](bool ok) {
                        auto& panel=d_.op==9?downPanels_[index]:panels_[index]; --panel.inFlight;
                        if (!ok) { finish(false); return; }
                        if (panel.nextRow==panel.rows && !panel.inFlight) panel.draining=false;
                    });
            }
        }
    }
    uint32_t configuredLayers() const {
        const char* value=std::getenv("GOLEM_DECODER_LAYERS");
        if (!value || !*value) return 0;
        char* end=nullptr; const unsigned long parsed=std::strtoul(value,&end,10);
        return end && *end==0 && parsed<=32 ? static_cast<uint32_t>(parsed) : 0;
    }
    uint32_t totalOutputInFlight() const { const auto* p=d_.op==9?downPanels_:panels_; return p[0].inFlight+p[1].inFlight; }
    uint64_t downOutputGM(uint32_t index) const {
        return gm_->getBaseAddr()+DecoderMemoryLayout::decoder_down_output_gm+index*DecoderMemoryLayout::decoder_panel_bytes;
    }
    bool startDown(Task resume) {
        for (uint32_t i=0;i<2;++i) {
            if (!downReady_[i] || panels_[i].firstTile!=downNextK_) continue;
            downActive_=true;downReady_[i]=false;downConsumer_=i;downResume_=std::move(resume);
            downN_=downK_=downWeightSlot_=0;
            for (uint32_t i=0;i<DecoderMemoryLayout::decoder_prefetch_slots;++i) downWeightReads_[i]=BackgroundRead{};
            downInputRead_=BackgroundRead{};
            if (tileMajorFfn()) { defer([this] { downLoadWeight(); });return true; }
            const auto& p=panels_[i];const size_t bytes=uint64_t(p.rows)*p.tiles*128;
            readLocal(panelGM(i),bytes,[this](const std::vector<uint8_t>& raw) {
                const auto& p=panels_[downConsumer_];std::vector<uint8_t> packed(raw.size());
                // Consume gated row-major output through real GM reads, then
                // reuse the now-dead up slot for K-major down input.
                for (uint32_t k=0;k<p.tiles;++k) for (uint32_t r=0;r<p.rows;++r)
                    std::copy_n(raw.begin()+(uint64_t(r)*p.tiles+k)*128,128,
                        packed.begin()+(uint64_t(k)*p.rows+r)*128);
                writeLocal(upPanelGM(downConsumer_),packed,[this] { downLoadWeight(); });
            });
            return true;
        }
        return false;
    }
    void waitDownRow() {
        if(pipelined()){
            if(downActive_ || downReady_[0] || downReady_[1] || downTask_ ||
               downNextK_!=d_.output_dim/64 || ffnState_!=FfnState::Idle ||
               ffnReady_[0] || ffnReady_[1] || cycle_<downPartialReady_){defer([this]{waitDownRow();});return;}
            row_+=256;defer([this]{loadRows();});return;
        }
        if (startDown([this] { waitDownRow(); })) return;
        if (downNextK_!=d_.output_dim/64 || ffnState_!=FfnState::Idle ||
            ffnReady_[0] || ffnReady_[1] || cycle_<downPartialReady_) {
            defer([this] { waitDownRow(); });return;
        }
        row_+=256;defer([this] { loadRows(); });
    }
    uint64_t downWeightAddress(uint32_t n,uint32_t k) const {
        return (1+uint64_t(n%4))*d_.node_stride+d_.down_weights_offset+
            (uint64_t(n/4)*(d_.output_dim/64)+panels_[downConsumer_].firstTile+k)*8192;
    }
    void beginDownWeightRead(uint32_t slot,uint32_t n,uint32_t k) {
        auto& r=downWeightReads_[slot];
        if (r.busy || r.ready || r.pending) { finish(false);return; }
        r.tile=n;r.k=k;r.pending=true;r.phase=1;
        r.weight=true;r.submitted=cycle_;r.source=downWeightAddress(n,k);
        r.address=gm_->getBaseAddr()+DecoderMemoryLayout::decoder_down_weight_gm+uint64_t(slot)*8192;
        r.data.resize(8192);hbmRead_+=8192;++prefetchRequests_;
        gm_->dma_read_from_host_to_globalmem(downWeightAddress(n,k),8192,r.address,[this,slot](bool ok) {
            auto& read=downWeightReads_[slot];read.pending=false;read.phase=2;read.dmaDone=cycle_;
            if (!ok) { finish(false);return; }read.busy=true;
        });
    }
    bool nextDown(uint32_t n,uint32_t k,uint32_t& nextN,uint32_t& nextK) const {
        if (reuseDownInput()) {
            nextN=n+1;nextK=k;
            if (nextN<d_.input_dim/64) return true;
            nextN=0;nextK=k+1;return nextK<panels_[downConsumer_].tiles;
        }
        nextN=n;nextK=k+1;
        if (nextK<panels_[downConsumer_].tiles) return true;
        nextN=n+1;nextK=0;
        return nextN<d_.input_dim/64;
    }
    bool hasDownRead(uint32_t n,uint32_t k) const {
        for (uint32_t i=0;i<prefetchSlots_;++i) {
            const auto& r=downWeightReads_[i];
            if ((r.pending||r.busy||r.ready) && r.tile==n && r.k==k) return true;
        }
        return false;
    }
    void scheduleDownPrefetch(uint32_t n,uint32_t k) {
        uint32_t slot=downWeightSlot_,nextN=n,nextK=k;
        for (uint32_t i=0;i+1<prefetchSlots_;++i) {
            if (!nextDown(nextN,nextK,nextN,nextK)) return;
            if (hasDownRead(nextN,nextK)) { slot=(slot+1)%prefetchSlots_;continue; }
            auto& r=downWeightReads_[slot];
            if (r.pending||r.busy||r.ready) return;
            beginDownWeightRead(slot,nextN,nextK);
            slot=(slot+1)%prefetchSlots_;
        }
    }
    void downLoadWeight() {
        if (allGemmWindow_) { startDownWindow(); return; }
        const bool final=panels_[downConsumer_].firstTile+panels_[downConsumer_].tiles==d_.output_dim/64;
        if (final && !downOutputCount_ && (downPanels_[downOutputIndex_].draining || downPanels_[downOutputIndex_].epilogue)) {
            defer([this] { downLoadWeight(); });return;
        }
        if (prefetchEnabled_) {
            auto& r=downWeightReads_[downWeightSlot_];
            if (!r.busy && !r.ready && !r.pending) beginDownWeightRead(downWeightSlot_,downN_,downK_);
            if (!r.ready) {
                ++prefetchWaitCycles_;
                if (r.phase==1) ++prefetchDmaWaitCycles_;
                else if (r.phase==2 || r.busy) ++prefetchGmWaitCycles_;
                else ++prefetchProgramWaitCycles_;
                defer([this] { downLoadWeight(); });return;
            }
            if (r.tile!=downN_ || r.k!=downK_) { finish(false);return; }
            traceWeight(r,true);operand_=decode(r.data);r=BackgroundRead{};++prefetchHits_;
            downWeightSlot_=(downWeightSlot_+1)%prefetchSlots_;
            scheduleDownPrefetch(downN_,downK_);
            if (!reuseDownInput() || !downN_) {
                downInputRead_=BackgroundRead{};
                downInputRead_.address=downInputGM()+uint64_t(downK_)*activeRows_*128;
                downInputRead_.data.resize(activeRows_*128);downInputRead_.busy=true;
            }
            defer([this] { downProgramWeight(); });
        } else loadHBM(downWeightAddress(downN_,downK_),
            gm_->getBaseAddr()+DecoderMemoryLayout::decoder_down_weight_gm,8192,[this] {
                readLocal(gm_->getBaseAddr()+DecoderMemoryLayout::decoder_down_weight_gm,8192,[this](const std::vector<uint8_t>& b) {
                    operand_=decode(b);downProgramWeight();
                });
            });
    }
    void downProgramWeight() {
        if (grouped()) { beginGroupedTile(true); return; }
        const uint64_t submitted=cycle_;
        if (!wcp_->programGemmMatrixGroupClassBankAsync(arrays_,0,operand_,2,
            AttentionClusterTrafficClass::ProjectionWeights,++tag_,cycle_,[this,submitted](bool ok,uint64_t) {
                weightProgramCycles_+=cycle_-submitted;
                if (!ok) { finish(false);return; }operand_.clear();
                if (reuseDownInput() && downN_) { downInputReady();return; }
                if (prefetchEnabled_) defer([this] { downConsumeInput(); });
                else readLocal(downInputGM()+uint64_t(downK_)*activeRows_*128,activeRows_*128,
                    [this](const std::vector<uint8_t>& b) { operand_=decode(b);downProgramInput(); });
            })) { ++prefetchProgramWaitCycles_; defer([this] { downProgramWeight(); }); }
    }
    void downConsumeInput() {
        if (!downInputRead_.ready) { defer([this] { downConsumeInput(); });return; }
        operand_=decode(downInputRead_.data);downInputRead_=BackgroundRead{};downProgramInput();
    }
    void downProgramInput() {
        if (!wcp_->programGemmInputScatterBankAsync(arrays_,0,operand_,2,
            AttentionClusterTrafficClass::ProjectionInputScatter,++tag_,cycle_,[this](bool ok,uint64_t) {
                if (!ok) { finish(false);return; }operand_.clear();
                downInputReady();
            })) defer([this] { downProgramInput(); });
    }
    void downInputReady() {
        const bool restore=reuseDownInput()?(panels_[downConsumer_].firstTile+downK_!=0):
            (!downK_ && panels_[downConsumer_].firstTile);
        if (restore) {
            if (cycle_<downPartialReady_) { defer([this] { downInputReady(); });return; }
            if (!wcp_->readProjectionPartial(uint64_t(downN_)*activeRows_*128,activeRows_*128,
                cycle_,downPartial_,downRestoreReady_)) { finish(false);return; }
            downCRead_+=downPartial_.size();defer([this] { downRestore(); });
        } else defer([this] { downLaunch(); });
    }

    void downRestore() {
        if (cycle_<downRestoreReady_) { defer([this] { downRestore(); });return; }
        const auto values=decode(downPartial_);
        if (!wcp_->writeGemmOutputGroupClassAsync(arrays_,values,2,AttentionClusterTrafficClass::ProjectionOutput,
            ++tag_,cycle_,[this](bool ok,uint64_t) {
                if (!ok) { finish(false);return; }downPartial_.clear();++downRestores_;defer([this] { downLaunch(); });
            })) defer([this] { downRestore(); });
    }
    void downLaunch() {
        const uint64_t launched=cycle_;
        remaining_=activeRows_;
        const uint64_t mode=panels_[downConsumer_].firstTile+downK_?1:0;
        if (!wcp_->launchGemmArrayGroupActiveBank(arrays_,0,mode,64,cycle_,[this,launched](uint32_t,uint64_t) {
            if (--remaining_) return;
            arrayComputeCycles_+=cycle_-launched;
            if (reuseDownInput()) defer([this] { downReadOutput(); });
            else if (++downK_<panels_[downConsumer_].tiles) defer([this] { downLoadWeight(); });
            else defer([this] { downReadOutput(); });
            })) { ++arrayWaitCycles_; defer([this] { downLaunch(); }); }
    }
    void downReadOutput() {
        if (groupRemaining_[0] || groupRemaining_[1]) { defer([this] { downReadOutput(); });return; }
        if (!wcp_->readGemmOutputGroupClassAsync(arrays_,2,AttentionClusterTrafficClass::ProjectionOutput,
            ++tag_,cycle_,[this](bool ok,uint64_t,const std::vector<double>& values) {
                if (!ok || values.size()!=activeRows_*64) { finish(false);return; }
                consumeDownOutput(values);
            })) defer([this] { downReadOutput(); });
    }
    void consumeDownOutput(const std::vector<double>& values) {
                const auto& p=panels_[downConsumer_];
                if (p.firstTile+(reuseDownInput()?downK_+1:p.tiles)<d_.output_dim/64) {
                    auto raw=encode(values);uint64_t ready;
                    if (!wcp_->writeProjectionPartial(uint64_t(downN_)*activeRows_*128,raw,cycle_,ready)) { finish(false);return; }
                    downCWrite_+=raw.size();downPartialReady_=std::max(downPartialReady_,ready);
                    window_.delayUntil(ready);
                    ++downN_;defer([this] { downNextOutput(); });
                } else {
                    if (!downOutputCount_) downOutputStart_=downN_;
                    const auto ready=[this] {
                        ++downOutputCount_;++downN_;
                        if (downOutputCount_==16 || downN_==d_.input_dim/64) downPrepareOutput();
                        else downNextOutput();
                    };
                    if (tileMajorFfn() && ffnLayout_==2)
                        writePanelTile(downOutputGM(downOutputIndex_),std::min(16u,d_.input_dim/64-downOutputStart_),
                            downOutputCount_,encode(values),ready);
                    else writeLocal(downOutputGM(downOutputIndex_)+uint64_t(downOutputCount_)*activeRows_*128,
                        encode(values),ready);
                }
    }
    // A software wave over two disjoint subsets of the EXISTING arrays. Each
    // array still has one operand context. Do not touch a subset until all its
    // previous MAC callbacks returned. The other subset may compute meanwhile.
    // Only one K tile's matrix/input snapshots are retained; the original GM
    // prefetch slot cannot be reused until both subsets have been submitted.
    bool grouped() const { return !(downActive_ && reuseDownInput()) && isFfn() && arrayGroups_==2 && activeRows_>32; }
    void beginGroupedTile(bool down) {
        groupDown_=down;groupIndex_=0;
        groupMatrix_=std::move(operand_);groupInput_.clear();
        if (down && !downK_ && panels_[downConsumer_].firstTile) {
            if (!wcp_->readProjectionPartial(uint64_t(downN_)*activeRows_*128,activeRows_*128,
                cycle_,downPartial_,downRestoreReady_)) { finish(false);return; }
            downCRead_+=downPartial_.size();defer([this] { groupRestore(); });
        } else defer([this] { groupProgram(); });
    }
    void groupRestore() {
        if (cycle_<downRestoreReady_) { defer([this] { groupRestore(); });return; }
        const auto values=decode(downPartial_);
        if (!wcp_->writeGemmOutputGroupClassAsync(arrays_,values,2,AttentionClusterTrafficClass::ProjectionOutput,
            ++tag_,cycle_,[this](bool ok,uint64_t) {
                if (!ok) { finish(false);return; }
                downPartial_.clear();++downRestores_;defer([this] { groupProgram(); });
            })) defer([this] { groupRestore(); });
    }
    std::vector<uint32_t> groupArrays() const {
        const uint32_t begin=groupIndex_*32,end=std::min(begin+32,activeRows_);
        return std::vector<uint32_t>(arrays_.begin()+begin,arrays_.begin()+end);
    }
    void groupProgram() {
        if (groupRemaining_[groupIndex_]) { ++arrayWaitCycles_;defer([this] { groupProgram(); });return; }
        const uint64_t submitted=cycle_;
        if (!wcp_->programGemmMatrixGroupClassBankAsync(groupArrays(),0,groupMatrix_,2,
            AttentionClusterTrafficClass::ProjectionWeights,++tag_,cycle_,[this,submitted](bool ok,uint64_t) {
                weightProgramCycles_+=cycle_-submitted;groupProgramCycles_+=cycle_-submitted;
                if (!ok) { finish(false);return; }
                defer([this] { groupReadInput(); });
            })) { ++prefetchProgramWaitCycles_;defer([this] { groupProgram(); }); }
    }
    void groupReadInput() {
        if (!groupInput_.empty()) { defer([this] { groupScatter(); });return; }
        if (prefetchEnabled_) {
            auto& read=groupDown_?downInputRead_:inputRead_;
            if (!read.ready) { defer([this] { groupReadInput(); });return; }
            groupInput_=decode(read.data);read=BackgroundRead{};
            defer([this] { groupScatter(); });
        } else {
            const uint64_t address=groupDown_?downInputGM()+uint64_t(downK_)*activeRows_*128:
                inputGM()+uint64_t(kt_)*activeRows_*128;
            readLocal(address,activeRows_*128,[this](const std::vector<uint8_t>& b) {
                groupInput_=decode(b);defer([this] { groupScatter(); });
            });
        }
    }
    void groupScatter() {
        const size_t begin=groupIndex_*32*64,end=std::min(begin+32*64,groupInput_.size());
        const std::vector<double> input(groupInput_.begin()+begin,groupInput_.begin()+end);
        if (!wcp_->programGemmInputScatterBankAsync(groupArrays(),0,input,2,
            AttentionClusterTrafficClass::ProjectionInputScatter,++tag_,cycle_,[this](bool ok,uint64_t) {
                if (!ok) { finish(false);return; }defer([this] { groupLaunch(); });
            })) defer([this] { groupScatter(); });
    }
    void groupLaunch() {
        const auto ids=groupArrays();const uint32_t group=groupIndex_,rows=ids.size();
        const uint64_t launched=cycle_,mode=groupDown_?(panels_[downConsumer_].firstTile+downK_?1:0):(kt_?1:0);
        groupRemaining_[group]=rows;
        if (!wcp_->launchGemmArrayGroupActiveBank(ids,0,mode,64,cycle_,[this,group,rows,launched](uint32_t,uint64_t) {
                if (--groupRemaining_[group]) return;
                arrayComputeCycles_+=cycle_-launched;
                groupComputeArrayCycles_+=(cycle_-launched)*rows;
                // No continuation here: only the main stream advances K.
            })) {
            groupRemaining_[group]=0;++arrayWaitCycles_;defer([this] { groupLaunch(); });return;
        }
        ++groupLaunches_;
        if (!groupIndex_) { groupIndex_=1;defer([this] { groupProgram(); });return; }
        groupMatrix_.clear();groupInput_.clear();
        if (groupDown_) {
            if (++downK_<panels_[downConsumer_].tiles) defer([this] { downLoadWeight(); });
            else defer([this] { downReadOutput(); });
        } else {
            if (++kt_<d_.input_dim/64) defer([this] { loadWeight(); });
            else defer([this] { readOutput(); });
        }
    }
    void downPrepareOutput() {
        const uint32_t index=downOutputIndex_,tiles=downOutputCount_,rows=activeRows_;
        const uint64_t address=downOutputGM(index);
        if (tileMajorFfn() && ffnLayout_==2) { downOutputPrepared(index,tiles,rows);return; }
        readLocal(address,uint64_t(rows)*tiles*128,[this,index,tiles,rows,address](const std::vector<uint8_t>& raw) {
            std::vector<uint8_t> packed(raw.size());
            for (uint32_t n=0;n<tiles;++n) for (uint32_t r=0;r<rows;++r)
                std::copy_n(raw.begin()+(uint64_t(n)*rows+r)*128,128,packed.begin()+(uint64_t(r)*tiles+n)*128);
            writeLocal(address,packed,[this,index,tiles,rows] { downOutputPrepared(index,tiles,rows); });
        });
    }
    void downOutputPrepared(uint32_t index,uint32_t tiles,uint32_t rows) {
                auto& p=downPanels_[index];p.row=row_;p.rows=rows;p.firstTile=downOutputStart_;p.tiles=tiles;
                p.epilogue=d_.residual_offset!=0;p.draining=!p.epilogue;p.nextRow=p.inFlight=0;
                downOutputCount_=0;downOutputIndex_=1-index;downNextOutput();
    }

    void downNextOutput() {
        if (windowDownSink_) { windowDownSink_=false; window_.resumeOutput(); return; }
        if (downN_<d_.input_dim/64) {
            if (!reuseDownInput()) downK_=0;
            defer([this] { downLoadWeight(); });return;
        }
        if (reuseDownInput() && downK_+1<panels_[downConsumer_].tiles) {
            ++downK_;downN_=0;defer([this] { downLoadWeight(); });return;
        }
        downNextK_=panels_[downConsumer_].firstTile+panels_[downConsumer_].tiles;
        ++downPasses_;downActive_=false;
        auto resume=std::move(downResume_);defer(std::move(resume));
    }
    void beginFfnChunk() {
        const auto& p=panels_[ffnConsumer_];
        const size_t width=p.tiles*64, total=uint64_t(p.rows)*width;
        ffnElements_=std::min(total-ffnOffset_,(8192/width)*width);
        ffnGateRead_=BackgroundRead{}; ffnUpRead_=BackgroundRead{};
        ffnGateRead_.address=panelGM(ffnConsumer_)+ffnOffset_*2;
        ffnUpRead_.address=upPanelGM(ffnConsumer_)+ffnOffset_*2;
        ffnGateRead_.data.resize(ffnElements_*2); ffnUpRead_.data.resize(ffnElements_*2);
        ffnGateRead_.busy=ffnUpRead_.busy=true;
        ffnState_=FfnState::Reading;
    }
    void beginResidualChunk() {
        const auto& p=downPanels_[residualPanel_];
        const uint32_t width=p.tiles*64;
        residualRows_=std::min(p.rows-residualRow_,8192u/(width*2));
        const size_t bytes=uint64_t(residualRows_)*width*2;
        residualRead_=BackgroundRead{};residualDownRead_=BackgroundRead{};
        residualRead_.address=gm_->getBaseAddr()+DecoderMemoryLayout::decoder_epilogue_gm;
        residualRead_.data.resize(bytes);
        residualDownRead_.address=downOutputGM(residualPanel_)+uint64_t(residualRow_)*width*2;
        residualDownRead_.data.resize(bytes);residualDownRead_.busy=true;
        residualPending_=residualRows_;residualState_=ResidualState::Reading;
        if(chainJob_==4){
            residualRead_.busy=false;residualCacheNext_=residualPending_=0;
            chainReadSaved_+=bytes;return;
        }
        for (uint32_t r=0;r<residualRows_;++r) {
            const uint64_t source=(d_.manager+1)*d_.node_stride+d_.residual_offset+
                ((uint64_t(p.row)+residualRow_+r)*d_.input_dim+p.firstTile*64)*2;
            hbmRead_+=width*2;residualBytes_+=width*2;
            gm_->dma_read_from_host_to_globalmem(source,width*2,residualRead_.address+uint64_t(r)*width*2,
                [this](bool ok) {
                    if (!ok) { finish(false);return; }
                    if (!--residualPending_) residualRead_.busy=true;
                });
        }
    }
    void pumpResidual() {
        if (residualState_!=ResidualState::Idle && wcp_->isBusy()) ++residualOverlap_;
        if(chainJob_==4 && residualState_==ResidualState::Reading && !residualRead_.ready){
            const auto& p=downPanels_[residualPanel_];const size_t bytes=p.tiles*128;
            while(residualCacheNext_<residualRows_ && residualPending_<writeWindow_){
                const uint32_t r=residualCacheNext_;
                const uint64_t address=chainGM()+((uint64_t(p.row-d_.lane*64)+residualRow_+r)*d_.input_dim+p.firstTile*64)*2;
                ++residualCacheNext_;++residualPending_;
                if(!gm_->localReadAsync(address,bytes,LocalMemoryClient::RoCC,++tag_,[this,r,bytes](bool ok,uint64_t,const std::vector<uint8_t>& b){
                    --residualPending_;if(!ok || b.size()!=bytes){finish(false);return;}
                    std::copy(b.begin(),b.end(),residualRead_.data.begin()+r*bytes);localRead_+=bytes;
                    if(residualCacheNext_==residualRows_ && !residualPending_)residualRead_.ready=true;
                })){--residualCacheNext_;--residualPending_;break;}
            }
        }
        pumpRead(residualRead_);pumpRead(residualDownRead_);
        switch (residualState_) {
        case ResidualState::Idle:
            for (uint32_t i=0;i<2;++i) if (downPanels_[i].epilogue) {
                residualPanel_=i;residualRow_=0;beginResidualChunk();break;
            }
            break;
        case ResidualState::Reading:
            if (residualRead_.ready && residualDownRead_.ready) {
                residualInput_=decode(residualDownRead_.data);residualCoefficients_=decode(residualRead_.data);
                residualRead_=BackgroundRead{};residualDownRead_=BackgroundRead{};
                residualState_=ResidualState::Issue;
            }
            break;
        case ResidualState::Issue: {
            SFUVectorRequest request{};request.op=SFUVectorOp::Add;
            request.input=&residualInput_;request.coefficients=&residualCoefficients_;
            request.rowWidth=downPanels_[residualPanel_].tiles*64;
            if (sfu_->issueVectorOp(request,&residualResult_)) {
                residualInput_.clear();residualCoefficients_.clear();residualState_=ResidualState::Wait;
            }
            break;
        }
        case ResidualState::Wait:
            if (residualResult_.completed && tick_>=residualResult_.readyTick) {
                residualWrite_=encode(residualResult_.values);residualResult_.values.clear();
                residualWriteOffset_=0;residualState_=ResidualState::Writing;
            }
            break;
        case ResidualState::Writing: {
            if (residualWritePending_) break;
            const auto& p=downPanels_[residualPanel_];const size_t rowBytes=p.tiles*128;
            const size_t len=std::min({gm_->localMaxRequestBytes(),residualWrite_.size()-residualWriteOffset_,
                layerProducer_ && p.tiles*64<d_.input_dim?rowBytes-residualWriteOffset_%rowBytes:residualWrite_.size()});
            const uint64_t address=layerProducer_?chainGM()+
                ((uint64_t(p.row-d_.lane*64)+residualRow_+residualWriteOffset_/rowBytes)*d_.input_dim+p.firstTile*64)*2+
                residualWriteOffset_%rowBytes:downOutputGM(residualPanel_)+uint64_t(residualRow_)*rowBytes+residualWriteOffset_;
            std::vector<uint8_t> bytes(residualWrite_.begin()+residualWriteOffset_,residualWrite_.begin()+residualWriteOffset_+len);
            residualWritePending_=true;
            if (!gm_->localWriteAsync(address,bytes,LocalMemoryClient::RoCC,++tag_,[this,len](bool ok,uint64_t) {
                residualWritePending_=false;
                if (!ok) { finish(false);return; }
                localWrite_+=len;if(layerProducer_)layerBytes_+=len;residualWriteOffset_+=len;
                if (residualWriteOffset_!=residualWrite_.size()) return;
                residualWrite_.clear();residualRow_+=residualRows_;++residualChunks_;
                auto& p=downPanels_[residualPanel_];
                if (residualRow_<p.rows) beginResidualChunk();
                else { p.epilogue=false;p.draining=true;residualState_=ResidualState::Idle; }
            })) residualWritePending_=false;
            break;
        }
        }
    }
    void pumpFfn() {
        if (!active_) return;
        if(streamFfn_){pumpFfnStream();return;}
        if ((ffnState_==FfnState::WaitSilu || ffnState_==FfnState::WaitMultiply) && wcp_->isBusy()) ++ffnOverlapCycles_;
        switch (ffnState_) {
        case FfnState::Idle:
            if (ffnReady_[ffnConsumer_]) {
                ffnReady_[ffnConsumer_]=false; ffnOffset_=0; beginFfnChunk();
            }
            break;
        case FfnState::Reading:
            if (ffnGateRead_.ready && ffnUpRead_.ready) {
                ffnInput_=decode(ffnGateRead_.data); ffnCoefficients_=decode(ffnUpRead_.data);
                ffnGateRead_=BackgroundRead{}; ffnUpRead_=BackgroundRead{};
                ffnState_=FfnState::IssueSilu;
            }
            break;
        case FfnState::IssueSilu:
        case FfnState::IssueMultiply: {
            const bool silu=ffnState_==FfnState::IssueSilu;
            SFUVectorRequest r{}; r.input=&ffnInput_; r.coefficients=&ffnCoefficients_;
            r.rowWidth=panels_[ffnConsumer_].tiles*64;
            r.op=silu?SFUVectorOp::Silu:SFUVectorOp::Multiply;
            if (sfu_->issueVectorOp(r,&ffnResult_)) {
                ffnInput_.clear(); if (!silu) ffnCoefficients_.clear();
                ffnState_=silu?FfnState::WaitSilu:FfnState::WaitMultiply;
            }
            break;
        }
        case FfnState::WaitSilu:
            if (ffnResult_.completed && tick_>=ffnResult_.readyTick) {
                // SFU output has already crossed the same FP16 boundary as
                // the standalone SiLU tensor. No float-only shortcut.
                ffnInput_=std::move(ffnResult_.values); ffnState_=FfnState::IssueMultiply;
            }
            break;
        case FfnState::WaitMultiply:
            if (ffnResult_.completed && tick_>=ffnResult_.readyTick) {
                ffnWrite_=encode(ffnResult_.values); ffnResult_.values.clear();
                ffnWriteOffset_=0; ffnState_=FfnState::Writing;
            }
            break;
        case FfnState::Writing: {
            if (ffnWritePending_) break;
            const size_t len=std::min(gm_->localMaxRequestBytes(),ffnWrite_.size()-ffnWriteOffset_);
            std::vector<uint8_t> bytes(ffnWrite_.begin()+ffnWriteOffset_,ffnWrite_.begin()+ffnWriteOffset_+len);
            ffnWritePending_=true;
            if (!gm_->localWriteAsync(panelGM(ffnConsumer_)+ffnOffset_*2+ffnWriteOffset_,bytes,
                LocalMemoryClient::RoCC,++tag_,[this,len](bool ok,uint64_t) {
                    ffnWritePending_=false;
                    if (!ok) { finish(false); return; }
                    localWrite_+=len; ffnWriteOffset_+=len;
                    if (ffnWriteOffset_!=ffnWrite_.size()) return;
                    ffnWrite_.clear(); ffnOffset_+=ffnElements_; ++ffnChunks_;
                    auto& p=panels_[ffnConsumer_];
                    if (ffnOffset_<uint64_t(p.rows)*p.tiles*64) beginFfnChunk();
                    else {
                        // Only the transformed gate slot becomes a DMA source.
                        // Producer waits for its complete ACK drain before reuse.
                        if (d_.op==9) downReady_[ffnConsumer_]=true;
                        else { p.draining=true; p.nextRow=p.inFlight=0; }
                        ++ffnConsumed_; ffnConsumer_=1-ffnConsumer_; ffnState_=FfnState::Idle;
                    }
                })) ffnWritePending_=false;
            break;
        }
        }
    }
    void pumpFfnStream() {
        if(ffnState_==FfnState::Idle){
            if(!ffnReady_[ffnConsumer_])return;
            ffnReady_[ffnConsumer_]=false;ffnOffset_=0;ffnState_=FfnState::Reading;
        }
        const auto& panel=panels_[ffnConsumer_];
        const size_t width=panel.tiles*64,total=size_t(panel.rows)*width;
        uint32_t occupied=0;bool io=false,compute=false;
        for(auto& c:ffnStream_){
            if(c.state==FfnState::Idle && ffnOffset_<total){
                c=FfnChunk{};c.offset=ffnOffset_;
                c.elements=std::min(total-ffnOffset_,std::max(width,(size_t(ffnChunkLimit_)/width)*width));
                ffnOffset_+=c.elements;
                c.gate.address=panelGM(ffnConsumer_)+c.offset*2;
                c.up.address=upPanelGM(ffnConsumer_)+c.offset*2;
                c.gate.data.resize(c.elements*2);c.up.data.resize(c.elements*2);
                c.gate.busy=c.up.busy=true;c.state=FfnState::Reading;
            }
            pumpRead(c.gate);pumpRead(c.up);
            if(c.state!=FfnState::Idle)++occupied;
            io|=c.state==FfnState::Reading || c.state==FfnState::Writing;
            compute|=c.state==FfnState::WaitSilu || c.state==FfnState::WaitMultiply;
            switch(c.state){
            case FfnState::Idle:break;
            case FfnState::Reading:
                if(c.gate.ready && c.up.ready){
                    c.input=decode(c.gate.data);c.coefficients=decode(c.up.data);
                    c.gate=BackgroundRead{};c.up=BackgroundRead{};c.state=FfnState::IssueSilu;
                }break;
            case FfnState::IssueSilu:
            case FfnState::IssueMultiply:{
                const bool silu=c.state==FfnState::IssueSilu;
                SFUVectorRequest request{};request.input=&c.input;request.coefficients=&c.coefficients;
                request.rowWidth=width;request.op=silu?SFUVectorOp::Silu:SFUVectorOp::Multiply;
                if(sfu_->issueVectorOp(request,&c.result)){
                    c.input.clear();if(!silu)c.coefficients.clear();
                    c.state=silu?FfnState::WaitSilu:FfnState::WaitMultiply;
                }break;
            }
            case FfnState::WaitSilu:
                if(c.result.completed && tick_>=c.result.readyTick){
                    // Preserve exactly the standalone FP16 SiLU boundary.
                    c.input=std::move(c.result.values);c.state=FfnState::IssueMultiply;
                }break;
            case FfnState::WaitMultiply:
                if(c.result.completed && tick_>=c.result.readyTick){
                    c.bytes=encode(c.result.values);c.result.values.clear();c.state=FfnState::Writing;
                }break;
            case FfnState::Writing:{
                if(c.writePending)break;
                const size_t len=std::min(gm_->localMaxRequestBytes(),c.bytes.size()-c.writeOffset);
                const uint64_t address=panelGM(ffnConsumer_)+c.offset*2+c.writeOffset;
                std::vector<uint8_t> bytes(c.bytes.begin()+c.writeOffset,c.bytes.begin()+c.writeOffset+len);
                c.writePending=true;FfnChunk* chunk=&c;
                if(!gm_->localWriteAsync(address,bytes,LocalMemoryClient::RoCC,++tag_,[this,chunk,len](bool ok,uint64_t){
                    chunk->writePending=false;if(!ok){finish(false);return;}
                    localWrite_+=len;chunk->writeOffset+=len;
                    if(chunk->writeOffset==chunk->bytes.size()){
                        chunk->bytes.clear();chunk->state=FfnState::Idle;++ffnChunks_;
                    }
                }))c.writePending=false;
                break;
            }
            }
        }
        ffnStreamHighWater_=std::max<uint64_t>(ffnStreamHighWater_,occupied);
        if(io && compute)++ffnStreamOverlap_;
        if(compute && wcp_->isBusy())++ffnOverlapCycles_;
        if(ffnOffset_==total && ffnStream_[0].state==FfnState::Idle && ffnStream_[1].state==FfnState::Idle){
            if(d_.op==9)downReady_[ffnConsumer_]=true;
            else{auto& p=panels_[ffnConsumer_];p.draining=true;p.nextRow=p.inFlight=0;}
            ++ffnConsumed_;ffnConsumer_=1-ffnConsumer_;ffnState_=FfnState::Idle;
        }
    }
    void loadVector() {
        siluMultiplySecond_=false;
        fusedNormPhase_=false;
        // At most 8192 FP16 elements fit the bounded vector request.
        vectorRows_=std::min(activeRows_-(vectorRow_-row_), std::max(1u,8192u/d_.input_dim));
        const size_t bytes=vectorRows_*d_.input_dim*2;
        const auto consume=[this,bytes](uint64_t address) {
            readLocal(address,bytes,[this](const std::vector<uint8_t>& b) {
                vectorInput_=decode(b);
                if (d_.op==4) { vectorCoefficients_.clear(); issueVector(); return; }
                const bool norm=d_.op==3;
                const size_t bytes=(norm?d_.input_dim:vectorInput_.size())*2;
                const auto coefficients=[this,bytes,norm] {
                    readLocal(weightGM(),bytes,[this](const std::vector<uint8_t>& b) {
                        vectorCoefficients_=decode(b); issueVector();
                    });
                    if(norm)gammaLoaded_=true;
                };
                if(layerResidualConsumer_){
                    const uint64_t residual=chainGM()+uint64_t(vectorRow_-row_)*d_.input_dim*2;
                    readLocal(residual,bytes,[this,bytes](const std::vector<uint8_t>& b){
                        layerResidualSaved_+=bytes;vectorCoefficients_=decode(b);issueVector();
                    });
                } else if(norm && reuseGamma_ && gammaLoaded_)coefficients();
                else {
                    if(fusedAddNorm_ && !norm)gammaLoaded_=false;
                    loadHBM(d_.input1_addr+(norm?0:uint64_t(vectorRow_)*d_.input_dim*2),weightGM(),bytes,coefficients);
                }
            });
        };
        if(layerConsumer_){layerSaved_+=bytes;consume(chainGM()+uint64_t(vectorRow_-row_)*d_.input_dim*2);}
        else if(chainJob_==2){
            chainReadSaved_+=bytes;consume(chainWoGM()+uint64_t(vectorRow_-row_)*d_.input_dim*2);
        } else if(chainJob_==3){
            chainReadSaved_+=bytes;consume(chainGM()+uint64_t(vectorRow_-row_)*d_.input_dim*2);
        } else loadHBM(d_.input_addr+uint64_t(vectorRow_)*d_.input_dim*2,inputGM(),bytes,[this,consume]{consume(inputGM());});
    }
    void issueVector() {
        SFUVectorRequest r{}; r.input=&vectorInput_; r.coefficients=&vectorCoefficients_;
        r.rowWidth=d_.input_dim; r.epsilon=1.0e-6f;
        r.op=fusedNormPhase_?SFUVectorOp::RmsNorm:d_.op==2?SFUVectorOp::Add:d_.op==3?SFUVectorOp::RmsNorm:
            (d_.op==4 || (d_.op==7 && !siluMultiplySecond_))?SFUVectorOp::Silu:SFUVectorOp::Multiply;
        if (!sfu_->issueVectorOp(r,&vectorResult_)) {
            if (d_.op==3 && !chainJob_) finish(false);
            else defer([this] { issueVector(); }); // finite request queue: retry
            return;
        }
        vectorInput_.clear();
        if (d_.op!=7 || siluMultiplySecond_) vectorCoefficients_.clear();
        vectorWaiting_=true;
        if(fusedAddNorm_)fusedVectorSubmit_=tick_;
    }
    void loadFusedNormGamma() {
        fusedNormPhase_=true;
        const size_t bytes=d_.input_dim*2;
        const auto consume=[this,bytes]{
            readLocal(weightGM(),bytes,[this](const std::vector<uint8_t>& b){
                vectorCoefficients_=decode(b);gammaLoaded_=true;
                const size_t saved=vectorRows_*d_.input_dim*2;
                fusedNormReadSaved_+=saved;chainReadSaved_+=saved;issueVector();
            });
        };
        if(reuseGamma_ && gammaLoaded_)consume();
        else loadHBM((d_.manager+1)*d_.node_stride+d_.weights_offset,weightGM(),bytes,consume);
    }
    void advanceVectorRows(){
        vectorRow_+=vectorRows_;
        if(vectorRow_<row_+activeRows_)loadVector();else{row_+=256;loadRows();}
    }
    void storeVector() {
        const size_t bytes=vectorRows_*d_.input_dim*2;
        if(chainJob_==3 || fusedAddNorm_){chainWriteSaved_+=bytes;advanceVectorRows();return;}
        storeHBM(outGM(),d_.output_addr+uint64_t(vectorRow_)*d_.input_dim*2,bytes,[this]{advanceVectorRows();});
    }
    void cacheNormChunk(const std::vector<uint8_t>& bytes){
        localBuffer_=bytes;normWriteNext_=normWriteDone_=normWritePending_=0;
        if(fusedAddNorm_){
            // Repack only the existing bounded SFU result. Consecutive rows of
            // each K tile share a timed request, within the existing GM limit.
            for(size_t k=0;k<d_.input_dim/64;++k)for(size_t r=0;r<vectorRows_;++r)
                std::copy_n(bytes.begin()+(r*d_.input_dim+k*64)*2,128,
                            localBuffer_.begin()+(k*vectorRows_+r)*128);
        }
        defer([this]{pumpNormCache();});
    }
    void pumpNormCache(){
        const size_t width=d_.input_dim;
        if(normWriteDone_==localBuffer_.size()){
            localBuffer_.clear();storeVector();return;
        }
        while(normWriteNext_<localBuffer_.size() && normWritePending_<writeWindow_){
            const size_t offset=normWriteNext_,element=offset/2,r=element/width,k=(element%width)/64;
            const size_t span=fusedAddNorm_?vectorRows_*128:128;
            const size_t len=std::min({span-offset%span,gm_->localMaxRequestBytes(),localBuffer_.size()-offset});
            const uint64_t address=(chainJob_==3 || fusedAddNorm_?inputGM():gm_->getBaseAddr()+DecoderMemoryLayout::decoder_norm_handoff_gm)+
                (fusedAddNorm_?(offset/span*activeRows_+vectorRow_-row_)*128+offset%span:
                 (k*activeRows_+(vectorRow_-row_)+r)*128+offset%128);
            std::vector<uint8_t> bytes(localBuffer_.begin()+offset,localBuffer_.begin()+offset+len);
            normWriteNext_+=len;++normWritePending_;
            if(!gm_->localWriteAsync(address,bytes,LocalMemoryClient::RoCC,++tag_,[this,len](bool ok,uint64_t){
                --normWritePending_;if(!ok){finish(false);return;}
                normWriteDone_+=len;normCacheWritten_+=len;localWrite_+=len;if(chainJob_==3 || fusedAddNorm_)chainLocalBytes_+=len;
            })){normWriteNext_-=len;--normWritePending_;break;}
        }
        defer([this]{pumpNormCache();});
    }
    void signal() {
        if (panels_[0].draining || panels_[1].draining || downPanels_[0].draining || downPanels_[1].draining ||
            downActive_ || downReady_[0] || downReady_[1] || cycle_<downPartialReady_ ||
            ffnReady_[0] || ffnReady_[1] || ffnState_!=FfnState::Idle ||
            downPanels_[0].epilogue || downPanels_[1].epilogue || residualState_!=ResidualState::Idle) {
            defer([this] { signal(); }); return;
        }
        if(sharedWeights_){
            if(!sharing_->release(d_.epoch)){defer([this]{signal();});return;}
            if(!sharedAccounted_){hbmRead_+=sharing_->hbmBytes(d_.epoch);
                localRead_+=sharing_->localReadBytes(d_.epoch);localWrite_+=sharing_->localWriteBytes(d_.epoch);
                sharing_->trace(d_.epoch);sharedAccounted_=true;}
        }
        if(layerProducer_ && !layerPublished_){
            const uint64_t expected=uint64_t(std::min(64u,d_.rows-d_.lane*64))*d_.input_dim*2;
            if(layerBytes_!=expected){finish(false);return;}
            writeLocal(gm_->getBaseAddr()+SoftwareLayerResidency::record,
                SoftwareLayerResidency::encode(d_.epoch,d_.output_addr,d_.rows,d_.input_dim,d_.lane,d_.manager),
                [this]{layerPublished_=true;defer([this]{signal();});});return;
        }
        if (!d_.completion_addr) { finish(true); return; }
        std::vector<uint8_t> b(8,0); const uint64_t value=d_.epoch ? d_.epoch : 1;
        std::memcpy(b.data(),&value,8);
        writeLocal(outGM(),b,[this] { storeHBM(outGM(),d_.completion_addr,8,[this] { finish(true); }); });
    }
    GlobalMemoryAPI* gm_; WorkerCommandProcessorAPI* wcp_; SFUAPI* sfu_;
    ReuseWindowGemm window_,downWindow_;
    bool blockPipeline_=false,pipelineProducerBlock_=false;uint32_t pipelinePolicy_=2;uint32_t arrayOwner_=0,arrayPreferred_=1;bool arrayWaiting_[2]{};
    Task downTask_;std::vector<uint8_t> pipelineWrite_;size_t pipelineWriteOffset_=0,pipelineWriteNext_=0;
    uint32_t pipelineWritePending_=0,pipelineWriteN_=0;
    uint64_t pipelineProducerTransfers_=0,pipelineDownTransfers_=0,pipelineBothActive_=0,pipelinePasses_=0,pipelineLeaseWait_=0;
    // Software residency metadata only; payload lives in existing timed GM.
    bool stageChain_=false;uint32_t chainState_=0,chainJob_=0,residualCacheNext_=0;
    bool addNormFusion_=false,fusedAddNorm_=false,fusedNormPhase_=false;
    uint64_t fusedNormChunks_=0,fusedNormReadSaved_=0;
    uint64_t fusedVectorSubmit_=0;
    bool layerResidency_=false,layerProducer_=false,layerConsumer_=false,layerPublished_=false;
    bool inputResidency_=false,layerResidualConsumer_=false,elideIntermediateOutput_=false,elideOutput_=false;uint64_t layerResidualSaved_=0;
    uint64_t layerBytes_=0,layerSaved_=0,elidedOutputBytes_=0;
    bool lmInputResidency_=false,lmInputValid_=false,useLmInput_=false;
    DecoderOpDesc lmInputDesc_{};uint64_t lmInputSaved_=0;
    DecoderOpDesc chainDesc_{};uint64_t chainH1Address_=0,chainNormAddress_=0;
    uint64_t chainReadSaved_=0,chainWriteSaved_=0,chainLocalBytes_=0;
    bool allGemmWindow_=false,windowDownSink_=false;
    bool reuseGamma_=false,gammaLoaded_=false;
    bool normHandoff_=false,cacheNorm_=false,useNormCache_=false,normCacheValid_=false;
    DecoderOpDesc normCacheDesc_{};
    uint64_t normCacheWritten_=0,normCacheReadSaved_=0;
    size_t normWriteNext_=0,normWriteDone_=0;uint32_t normWritePending_=0;
    ReuseWeightInterface* sharing_=nullptr;bool sharedWeights_=false,sharedAccounted_=false;
    bool downOwnedBuffer_=false,downActive_=false,downReady_[2]{};
    uint32_t downConsumer_=0,downNextK_=0,downN_=0,downK_=0,downWeightSlot_=0;
    uint32_t downOutputIndex_=0,downOutputCount_=0,downOutputStart_=0;
    uint64_t downPartialReady_=0,downRestoreReady_=0,downCRead_=0,downCWrite_=0,downRestores_=0,downPasses_=0;
    std::vector<uint8_t> downPartial_;Task downResume_;
    BackgroundRead downWeightReads_[DecoderMemoryLayout::decoder_prefetch_slots],downInputRead_;Panel downPanels_[2];
    ResidualState residualState_=ResidualState::Idle;
    BackgroundRead residualRead_,residualDownRead_;SFUVectorResult residualResult_{};
    uint32_t residualPanel_=0,residualRow_=0,residualRows_=0,residualPending_=0;
    size_t residualWriteOffset_=0;bool residualWritePending_=false;
    std::vector<double> residualInput_,residualCoefficients_;std::vector<uint8_t> residualWrite_;
    uint64_t residualChunks_=0,residualBytes_=0,residualOverlap_=0;
    DecoderOpDesc d_{}; Task next_,writeDone_; std::function<void(const std::vector<uint8_t>&)> readDone_;
    bool active_=false,done_=false,started_=false,vectorWaiting_=false;
    bool singleRow_=false;
    uint32_t lmParallel_=1,lmWidth_=0,lmIndex_=0;
    uint64_t lmLaunches_=0,lmArrayTiles_=0;
    bool siluMultiplySecond_=false;
    bool streamFfn_=false;uint32_t ffnChunkLimit_=8192;
    FfnChunk ffnStream_[2];uint64_t ffnStreamOverlap_=0,ffnStreamHighWater_=0;
    bool ffnUp_=false,ffnReady_[2]{},ffnWritePending_=false;
    FfnState ffnState_=FfnState::Idle;
    BackgroundRead ffnGateRead_,ffnUpRead_;
    uint32_t ffnPanelLimit_=16,ffnConsumer_=0;
    size_t ffnOffset_=0,ffnElements_=0,ffnWriteOffset_=0;
    std::vector<uint8_t> ffnWrite_;
    std::vector<double> ffnInput_,ffnCoefficients_; SFUVectorResult ffnResult_{};
    uint64_t ffnProduced_=0,ffnConsumed_=0,ffnChunks_=0,ffnOverlapCycles_=0,ffnSlotWaitCycles_=0;
    uint32_t activeRows_=64;
    Panel panels_[2];
    BackgroundRead weightReads_[DecoderMemoryLayout::decoder_prefetch_slots],inputRead_;
    uint32_t weightSlot_=0;
    bool prefetchEnabled_=false;
    uint32_t prefetchSlots_=2;
    uint64_t prefetchRequests_=0,prefetchHits_=0,prefetchWaitCycles_=0;
    uint64_t prefetchDmaWaitCycles_=0,prefetchGmWaitCycles_=0,prefetchProgramWaitCycles_=0,arrayWaitCycles_=0;
    bool weightTrace_=false;
    uint32_t ffnLayout_=0;bool downInputReuse_=false;
    size_t stridedNext_=0,stridedDone_=0,stridedPending_=0;uint64_t stridedStride_=0;
    uint32_t arrayGroups_=1,groupIndex_=0,groupRemaining_[2]{};
    bool groupDown_=false;
    std::vector<double> groupMatrix_,groupInput_;
    uint64_t groupLaunches_=0,groupProgramCycles_=0,groupComputeArrayCycles_=0,groupOverlapCycles_=0;
    uint64_t weightDmaCycles_=0,weightGmCycles_=0,weightProgramCycles_=0,arrayComputeCycles_=0;
    uint32_t panelLimit_=16,writeWindow_=8,panelIndex_=0,panelCount_=0,panelStart_=0;
    uint64_t outputRequests_=0,outputHighWater_=0;
    uint32_t embeddingRow_=0;
    std::vector<uint32_t> embeddingTokens_;
    uint64_t status_=0,cycle_=0,tick_=0,startTick_=0,tag_=0,localAddr_=0;
    uint64_t hbmRead_=0,hbmWrite_=0,localRead_=0,localWrite_=0;
    uint32_t row_=0,nt_=0,kt_=0,head_=0,remaining_=0,storeRow_=0,vectorRow_=0,vectorRows_=0;
    size_t localOffset_=0;
    std::vector<uint32_t> arrays_; std::vector<uint8_t> localBuffer_,gathered_;
    std::vector<double> operand_,vectorInput_,vectorCoefficients_; SFUVectorResult vectorResult_{};
};
} }
#endif
