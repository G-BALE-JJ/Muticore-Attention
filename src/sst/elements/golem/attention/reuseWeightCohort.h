#ifndef SST_GOLEM_REUSE_WEIGHT_COHORT_H
#define SST_GOLEM_REUSE_WEIGHT_COHORT_H
#include <sst/elements/golem/attention/reuseWeightInterface.h>
#include <sst/elements/golem/attention/decoderMemoryLayout.h>
#include <sst/elements/golem/globalmemory/globalmemory.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <deque>
#include <map>
#include <stdexcept>
#include <vector>
namespace SST { namespace Golem {
// Bounded software fetch sharing. Cache bytes live in existing, otherwise idle
// FFN input GM, not in an untimed host cache. Every fanout traverses the existing
// control NoC and every consumer lands through the real local GM write port.
class ReuseWeightCohort final : public ReuseWeightInterface {
    using Message=ControlTransportMessage;
    using Kind=ControlTransportMessageKind;
    struct Epoch {uint32_t joined=0,finished=0;bool finishSent=false,released=false;
        uint64_t requests=0,serverRequests=0,hits=0,hbm=0,read=0,written=0;};
    struct Slot {uint64_t source=0,age=0;uint32_t epoch=0;bool dma=false,reading=false,pending=false,ready=false;
        size_t offset=0;std::vector<uint8_t> bytes;std::vector<Message> waiters;};
    struct Landing {uint32_t epoch=0;uint64_t destination=0;size_t offset=0;bool pending=false;
        std::vector<uint8_t> bytes;std::function<void(bool)> done;};
public:
    ReuseWeightCohort(GlobalMemoryAPI* gm,uint32_t core,uint32_t width,uint32_t count=4)
        :gm_(gm),core_(core),width_(width),count_(count){
        if(core<4 || core>=20 || (width!=2 && width!=4) || (count!=4 && count!=8))
            throw std::invalid_argument("GEMM weight cohort");
        const uint32_t manager=(core-4)%4,lane=(core-4)/4;
        leader_=4+manager+(lane/width)*width*4;
        for(uint32_t i=0;i<width;++i)mask_|=1u<<(leader_+i*4);
        scratch_=gm_->getBaseAddr()+DecoderMemoryLayout::decoder_shared_weight_gm;
        if(scratch_+count_*8192>gm_->getBaseAddr()+gm_->getSize()-64)
            throw std::invalid_argument("GEMM weight scratch");
    }
    static bool accepts(Kind kind){return kind==Kind::GemmWeightJoin || kind==Kind::GemmWeightRequest ||
        kind==Kind::GemmWeightDelivery || kind==Kind::GemmWeightFinish || kind==Kind::GemmWeightRelease;}
    void begin(uint32_t epoch) override {
        auto& e=epochs_[epoch];if(e.requests || e.finishSent)throw std::logic_error("repeated cohort epoch");
        Message m{};m.kind=Kind::GemmWeightJoin;m.jobId=epoch;m.ownerCore=core_;send(leader_,std::move(m));
    }
    void fetch(uint32_t epoch,uint64_t source,uint64_t destination,std::function<void(bool)> done) override {
        if(landings_.size()>=4)throw std::logic_error("cohort consumer slot overflow");
        const auto token=++tag_;auto& l=landings_[token];l.epoch=epoch;l.destination=destination;l.done=std::move(done);
        ++epochs_[epoch].requests;
        Message m{};m.kind=Kind::GemmWeightRequest;m.jobId=epoch;m.tag=token;m.ownerCore=core_;
        m.inputAddr=source;m.outputAddr=destination;send(leader_,std::move(m));
    }
    bool release(uint32_t epoch) override {
        auto& e=epochs_[epoch];if(!landings_.empty())return false;
        if(!e.finishSent){e.finishSent=true;Message m{};m.kind=Kind::GemmWeightFinish;
            m.jobId=epoch;m.ownerCore=core_;send(leader_,std::move(m));}
        return e.released;
    }
    uint64_t hbmBytes(uint32_t epoch) const override {return epochs_.at(epoch).hbm;}
    uint64_t localReadBytes(uint32_t epoch) const override {return epochs_.at(epoch).read;}
    uint64_t localWriteBytes(uint32_t epoch) const override {return epochs_.at(epoch).written;}
    void trace(uint32_t epoch) const override {
        const auto& e=epochs_.at(epoch);
        std::printf("[GEMM_WEIGHT_SHARE] core=%u epoch=%u cohort=%u leader=%u slots=%u scratch=%llu requests=%llu server_requests=%llu cache_hits=%llu hbm_bytes=%llu gm_read=%llu gm_write=%llu released=%u\n",
            core_,epoch,width_,leader_,count_,(unsigned long long)scratch_,(unsigned long long)e.requests,
            (unsigned long long)e.serverRequests,(unsigned long long)e.hits,(unsigned long long)e.hbm,
            (unsigned long long)e.read,(unsigned long long)e.written,e.released?1:0);
    }
    void receive(const Message& m){
        if(!(mask_&(1u<<m.ownerCore)))throw std::logic_error("foreign cohort member");
        const uint32_t epoch=m.jobId;auto& e=epochs_[epoch];
        if(m.kind==Kind::GemmWeightDelivery){
            auto it=landings_.find(m.tag);
            if(it==landings_.end() || it->second.epoch!=epoch || !it->second.bytes.empty() || m.tensorPayload.size()!=8192)
                throw std::logic_error("invalid weight delivery");
            it->second.bytes=m.tensorPayload;return;
        }
        if(m.kind==Kind::GemmWeightRelease){e.released=true;return;}
        if(core_!=leader_)throw std::logic_error("request at nonleader");
        if(m.kind==Kind::GemmWeightJoin){e.joined|=1u<<m.ownerCore;return;}
        if(m.kind==Kind::GemmWeightFinish){e.finished|=1u<<m.ownerCore;return;}
        if(m.kind==Kind::GemmWeightRequest){
            ++e.serverRequests;
            if(requests_.size()>=width_*4)throw std::logic_error("cohort request queue overflow");
            requests_.push_back(m);return;
        }
        throw std::logic_error("unknown cohort message");
    }
    void tick(){
        pumpMessages();
        // Iterate a stable token list: synchronous stub callbacks may retire a landing.
        std::vector<uint64_t> tokens;for(const auto& kv:landings_)tokens.push_back(kv.first);
        for(auto token:tokens)pumpLanding(token);
        if(core_!=leader_)return;
        admit();for(uint32_t i=0;i<count_;++i)pumpSlot(i);
        for(auto& kv:epochs_){auto& e=kv.second;
            if(e.released || e.joined!=mask_ || e.finished!=mask_)continue;
            bool busy=!requests_.empty();
            for(uint32_t i=0;i<count_;++i)busy|=slots_[i].dma || slots_[i].reading || !slots_[i].waiters.empty();
            if(busy)continue;
            // All HBM/GM operations and all consumer landing ACKs have drained.
            e.released=true;
            for(uint32_t i=0;i<width_;++i){Message m{};m.kind=Kind::GemmWeightRelease;
                m.jobId=kv.first;m.ownerCore=leader_;send(leader_+i*4,std::move(m));}
        }
    }
private:
    void send(uint32_t destination,Message m){
        // At most 4 requests per member and bounded software fanout metadata.
        if(outgoing_.size()>width_*8+8)throw std::logic_error("cohort outgoing queue overflow");
        outgoing_.push_back({destination,std::move(m)});
    }
    void pumpMessages(){
        if(outgoing_.empty())return;
        auto& item=outgoing_.front();
        if(item.first==core_){auto m=std::move(item.second);outgoing_.pop_front();receive(m);}
        else if(gm_->sendControlMessage(item.first,item.second))outgoing_.pop_front();
    }
    void admit(){
        for(auto it=requests_.begin();it!=requests_.end();){
            if(epochs_[it->jobId].joined!=mask_){++it;continue;}
            uint32_t selected=count_;
            for(uint32_t i=0;i<count_;++i)if(slots_[i].source==it->inputAddr && slots_[i].epoch==it->jobId){
                selected=i;++epochs_[it->jobId].hits;break;}
            if(selected==count_){
                for(uint32_t i=0;i<count_;++i){const auto& s=slots_[i];
                    if(s.dma || s.reading || s.pending || !s.waiters.empty())continue;
                    if(selected==count_ || s.age<slots_[selected].age)selected=i;
                }
                if(selected==count_){++it;continue;}
                auto& s=slots_[selected];s=Slot{};s.source=it->inputAddr;s.epoch=it->jobId;s.dma=true;
                epochs_[s.epoch].hbm+=8192;
                gm_->dma_read_from_host_to_globalmem(s.source,8192,scratch_+selected*8192,
                    [this,selected](bool ok){if(!ok)throw std::runtime_error("shared HBM read failed");
                        slots_[selected].dma=false;slots_[selected].ready=true;});
            }
            auto& s=slots_[selected];s.age=++age_;s.waiters.push_back(*it);it=requests_.erase(it);
        }
    }
    void pumpSlot(uint32_t index){
        auto& s=slots_[index];if(s.dma || s.pending || s.waiters.empty())return;
        if(!s.reading){s.reading=true;s.offset=0;s.bytes.assign(8192,0);}
        const size_t len=std::min(gm_->localMaxRequestBytes(),8192-s.offset);s.pending=true;
        if(!gm_->localReadAsync(scratch_+index*8192+s.offset,len,LocalMemoryClient::RoCC,++tag_,
            [this,index,len](bool ok,uint64_t,const std::vector<uint8_t>& b){auto& s=slots_[index];s.pending=false;
                if(!ok || b.size()!=len)throw std::runtime_error("shared GM read failed");
                std::copy(b.begin(),b.end(),s.bytes.begin()+s.offset);s.offset+=len;epochs_[s.epoch].read+=len;
                if(s.offset!=8192)return;
                for(const auto& request:s.waiters){Message m{};m.kind=Kind::GemmWeightDelivery;m.ownerCore=leader_;
                    m.jobId=request.jobId;m.tag=request.tag;m.tensorPayload=s.bytes;send(request.ownerCore,std::move(m));}
                s.waiters.clear();s.bytes.clear();s.reading=false;
            }))s.pending=false;
    }
    void pumpLanding(uint64_t token){
        auto it=landings_.find(token);if(it==landings_.end())return;auto& l=it->second;
        if(l.pending || l.bytes.empty())return;
        const size_t len=std::min(gm_->localMaxRequestBytes(),8192-l.offset);
        std::vector<uint8_t> bytes(l.bytes.begin()+l.offset,l.bytes.begin()+l.offset+len);l.pending=true;
        if(!gm_->localWriteAsync(l.destination+l.offset,bytes,LocalMemoryClient::RoCC,++tag_,
            [this,token,len](bool ok,uint64_t){auto it=landings_.find(token);auto& l=it->second;l.pending=false;
                if(!ok)throw std::runtime_error("shared landing failed");
                l.offset+=len;epochs_[l.epoch].written+=len;if(l.offset!=8192)return;
                auto done=std::move(l.done);landings_.erase(it);done(true);
            }))l.pending=false;
    }
    GlobalMemoryAPI* gm_;uint32_t core_,width_,count_,leader_=0,mask_=0;uint64_t scratch_=0,tag_=0,age_=0;
    std::array<Slot,8> slots_{};std::map<uint32_t,Epoch> epochs_;std::map<uint64_t,Landing> landings_;
    std::deque<Message> requests_;std::deque<std::pair<uint32_t,Message>> outgoing_;
};
}}
#endif
