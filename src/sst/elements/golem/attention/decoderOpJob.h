#ifndef SST_GOLEM_DECODER_OP_JOB_H
#define SST_GOLEM_DECODER_OP_JOB_H
#include <sst/elements/golem/attention/decoderOpAbi.h>
#include <sst/elements/golem/attention/decoderMemoryLayout.h>
#include <sst/elements/golem/globalmemory/globalmemory.h>
#include <sst/elements/golem/workercmdproc/workercmdproc.h>
#include <sst/elements/golem/fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>
namespace SST { namespace Golem {
// Deliberately serial baseline: one operand context, bounded local scratch,
// timed GM accesses, existing WCP/arrays/SFU, and final HBM write ACKs.
class DecoderOpJob {
public:
    DecoderOpJob(GlobalMemoryAPI* gm, WorkerCommandProcessorAPI* wcp, SFUAPI* sfu)
        : gm_(gm), wcp_(wcp), sfu_(sfu) {
        for (uint32_t i = 0; i < 64; ++i) arrays_.push_back(i);
    }
    bool start(const DecoderOpDesc& d) {
        if (active_ || done_ || !gm_ || !wcp_ || !sfu_ || wcp_->isBusy() ||
            d.magic != GOLEM_DECODER_OP_MAGIC || d.size_bytes != sizeof(d) ||
            d.op < 1 || d.op > 6 || !d.rows ||
            d.lane > 3 || d.manager > 3 || !d.input_dim || (d.op!=6 && d.input_dim % 64) ||
            !d.output_dim || d.output_dim % 64 || (d.op!=6 && d.input_dim > 8192) || (d.op==6 && d.input_dim>1048576) ||
            d.output_dim > 8192 || !d.input_addr || !d.output_addr ||
            (d.op != 1 && d.op!=6 && d.input_dim != d.output_dim) ||
            (d.op != 1 && d.output_tile_begin) || (d.rows==1 && d.op==1 && d.head_dim) ||
            ((d.op == 2 || d.op == 3 || d.op == 5) && !d.input1_addr) ||
            ((d.op == 1 || d.op==6) && (!d.weights_offset || !d.node_stride ||
                (d.head_dim && (d.head_dim % 64 || d.input_dim % d.head_dim)))) ||
            gm_->getSize() < DecoderMemoryLayout::decoder_required_gm) return false;
        d_ = d; singleRow_=d.rows==1 && d.op==1;
        row_ = singleRow_ ? 0 : d.lane * 64; nt_ = kt_ = 0; active_ = true;
        status_ = 0; started_ = false; hbmRead_ = hbmWrite_ = localRead_ = localWrite_ = 0;
        defer([this] { loadRows(); });
        return true;
    }
    bool active() const { return active_; }
    bool complete() const { return done_; }
    uint64_t status() const { return status_; }
    void retire() { if (done_) { done_ = false; next_ = {}; } }
    void tick(uint64_t cycle, uint64_t tick) {
        cycle_ = cycle; tick_ = tick;
        if (!active_) return;
        if (!started_) {
            started_ = true; startTick_ = tick;
            std::printf("[DECODER_OP] event=start core=%u stage=%u op=%u tick=%llu K=%u N=%u epoch=%u\n",
                core(), d_.stage, d_.op, (unsigned long long)tick, d_.input_dim, d_.output_dim, d_.epoch);
        }
        if (vectorWaiting_ && vectorResult_.completed && tick >= vectorResult_.readyTick) {
            vectorWaiting_ = false;
            auto bytes = encode(vectorResult_.values);
            vectorResult_.values.clear();
            writeLocal(outGM(), bytes, [this] { storeVector(); });
        }
        if (next_) { auto f = std::move(next_); next_ = {}; f(); }
    }
private:
    using Task = std::function<void()>;
    uint32_t core() const { return 4 + d_.manager + d_.lane * 4; }
    uint64_t inputGM() const { return gm_->getBaseAddr() + DecoderMemoryLayout::decoder_input_gm; }
    uint64_t weightGM() const { return gm_->getBaseAddr() + DecoderMemoryLayout::decoder_weight_gm; }
    uint64_t outGM() const { return gm_->getBaseAddr() + DecoderMemoryLayout::decoder_output_gm; }
    void defer(Task f) { next_ = std::move(f); }
    void finish(bool ok) {
        status_ = ok ? 0 : 1; active_ = false; done_ = true; next_ = {};
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
        activeRows_=std::min(64u,d_.rows-row_);
        arrays_.resize(activeRows_);
        for (uint32_t i=0;i<activeRows_;++i) arrays_[i]=i;
        if (d_.op==6) { loadEmbeddingTokens(); return; }
        if (d_.op==1) {
            nt_=singleRow_ ? d_.manager+d_.lane*4 : 0; kt_=0;
            if (nt_>=d_.output_dim/64) { signal(); return; }
            if (d_.head_dim) { head_=0; gathered_.assign(activeRows_*d_.input_dim*2,0); gatherHead(); }
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
    void loadWeight() {
        const uint64_t tile=uint64_t(d_.output_tile_begin)+nt_;
        const uint64_t addr=(1+tile%4)*d_.node_stride+d_.weights_offset+
            ((tile/4)*(d_.input_dim/64)+kt_)*8192;
        loadHBM(addr,weightGM(),8192,[this] {
            readLocal(weightGM(),8192,[this](const std::vector<uint8_t>& b) {
                operand_=decode(b); defer([this] { programWeight(); });
            });
        });
    }
    void programWeight() {
        if (!wcp_->programGemmMatrixGroupClassBankAsync(arrays_,0,operand_,2,
            AttentionClusterTrafficClass::ProjectionWeights,++tag_,cycle_,[this](bool ok,uint64_t) {
                if (!ok) { finish(false); return; } operand_.clear();
                readLocal(inputGM()+uint64_t(kt_)*activeRows_*128,activeRows_*128,[this](const std::vector<uint8_t>& b) {
                    operand_=decode(b); defer([this] { programInput(); });
                });
            })) defer([this] { programWeight(); });
    }
    void programInput() {
        if (!wcp_->programGemmInputScatterBankAsync(arrays_,0,operand_,2,
            AttentionClusterTrafficClass::ProjectionInputScatter,++tag_,cycle_,[this](bool ok,uint64_t) {
                if (!ok) { finish(false); return; } operand_.clear(); defer([this] { launch(); });
            })) defer([this] { programInput(); });
    }
    void launch() {
        remaining_=activeRows_;
        if (!wcp_->launchGemmArrayGroupActiveBank(arrays_,0,kt_?1:0,64,cycle_,[this](uint32_t,uint64_t) {
            if (--remaining_) return;
            if (++kt_<d_.input_dim/64) defer([this] { loadWeight(); });
            else defer([this] { readOutput(); });
        })) defer([this] { launch(); });
    }
    void readOutput() {
        if (!wcp_->readGemmOutputGroupClassAsync(arrays_,2,AttentionClusterTrafficClass::ProjectionOutput,
            ++tag_,cycle_,[this](bool ok,uint64_t,const std::vector<double>& v) {
                if (!ok || v.size()!=activeRows_*64) { finish(false); return; }
                writeLocal(outGM(),encode(v),[this] { storeRow_=0; storeTileRow(); });
            })) defer([this] { readOutput(); });
    }
    void storeTileRow() {
        storeHBM(outGM()+storeRow_*128,d_.output_addr+(uint64_t(row_+storeRow_)*d_.output_dim+nt_*64)*2,
            128,[this] {
                if (++storeRow_<activeRows_) { storeTileRow(); return; }
                nt_+=singleRow_ ? 16 : 1;
                if (nt_<d_.output_dim/64) { kt_=0; loadWeight(); }
                else { row_+=256; loadRows(); }
            });
    }
    void loadVector() {
        // At most 8192 FP16 elements fit the bounded vector request.
        vectorRows_=std::min(activeRows_-(vectorRow_-row_), std::max(1u,8192u/d_.input_dim));
        const size_t bytes=vectorRows_*d_.input_dim*2;
        loadHBM(d_.input_addr+uint64_t(vectorRow_)*d_.input_dim*2,inputGM(),bytes,[this,bytes] {
            readLocal(inputGM(),bytes,[this](const std::vector<uint8_t>& b) {
                vectorInput_=decode(b);
                if (d_.op==4) { vectorCoefficients_.clear(); issueVector(); return; }
                const bool norm=d_.op==3;
                const size_t bytes=(norm?d_.input_dim:vectorInput_.size())*2;
                loadHBM(d_.input1_addr+(norm?0:uint64_t(vectorRow_)*d_.input_dim*2),weightGM(),bytes,[this,bytes] {
                    readLocal(weightGM(),bytes,[this](const std::vector<uint8_t>& b) {
                        vectorCoefficients_=decode(b); issueVector();
                    });
                });
            });
        });
    }
    void issueVector() {
        SFUVectorRequest r{}; r.input=&vectorInput_; r.coefficients=&vectorCoefficients_;
        r.rowWidth=d_.input_dim; r.epsilon=1.0e-6f;
        r.op=d_.op==2?SFUVectorOp::Add:d_.op==3?SFUVectorOp::RmsNorm:
            d_.op==4?SFUVectorOp::Silu:SFUVectorOp::Multiply;
        if (!sfu_->issueVectorOp(r,&vectorResult_)) {
            if (d_.op==3) finish(false);
            else defer([this] { issueVector(); }); // finite request queue: retry
            return;
        }
        vectorInput_.clear(); vectorCoefficients_.clear(); vectorWaiting_=true;
    }
    void storeVector() {
        const size_t bytes=vectorRows_*d_.input_dim*2;
        storeHBM(outGM(),d_.output_addr+uint64_t(vectorRow_)*d_.input_dim*2,bytes,[this] {
            vectorRow_+=vectorRows_;
            if (vectorRow_<row_+activeRows_) loadVector(); else { row_+=256; loadRows(); }
        });
    }
    void signal() {
        if (!d_.completion_addr) { finish(true); return; }
        std::vector<uint8_t> b(8,0); const uint64_t value=d_.epoch ? d_.epoch : 1;
        std::memcpy(b.data(),&value,8);
        writeLocal(outGM(),b,[this] { storeHBM(outGM(),d_.completion_addr,8,[this] { finish(true); }); });
    }
    GlobalMemoryAPI* gm_; WorkerCommandProcessorAPI* wcp_; SFUAPI* sfu_;
    DecoderOpDesc d_{}; Task next_,writeDone_; std::function<void(const std::vector<uint8_t>&)> readDone_;
    bool active_=false,done_=false,started_=false,vectorWaiting_=false;
    bool singleRow_=false;
    uint32_t activeRows_=64;
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
