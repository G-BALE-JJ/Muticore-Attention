// Test-only SST driver: real SFU scheduling and arithmetic with an in-memory GM.
#include <sst/core/component.h>
#include <sst/elements/golem/sfu/sfu.h>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
using namespace SST;
using namespace SST::Golem;

class SFUTestMemory : public GlobalMemoryAPI {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(SFUTestMemory, "golem", "SFUTestMemory",
        SST_ELI_ELEMENT_VERSION(1,0,0), "SFU regression memory", GlobalMemoryAPI)
    SFUTestMemory(ComponentId_t id, Params& p):GlobalMemoryAPI(id,p){}
    std::map<uint64_t,uint8_t> bytes;
    void wr_to_globalmem(uint64_t a,size_t n,const std::vector<uint8_t>& v) override {
        for(size_t i=0;i<n;++i)bytes[a+i]=v.at(i);
    }
    void rd_from_globalmem(uint64_t a,size_t n,std::vector<uint8_t>& v) override {
        v.resize(n);for(size_t i=0;i<n;++i)v[i]=bytes[a+i];
    }
    bool localReadAsync(uint64_t a,size_t n,LocalMemoryClient,uint64_t t,LocalReadCallback cb) override {
        std::vector<uint8_t> v;rd_from_globalmem(a,n,v);cb(true,t,v);return true;
    }
    bool localWriteAsync(uint64_t a,const std::vector<uint8_t>& v,LocalMemoryClient,uint64_t t,LocalWriteCallback cb) override {
        wr_to_globalmem(a,v.size(),v);cb(true,t);return true;
    }
    size_t localMaxRequestBytes() const override{return 256;}
    void wr_to_network(uint64_t,size_t,std::vector<uint8_t>&) override{assert(false);}
    void rd_to_network(uint64_t,size_t,uint64_t) override{assert(false);}
    void setBaseAddr(uint64_t) override{}
    uint64_t getBaseAddr() const override{return 0x1000;}
    uint64_t getSize() const override{return 1<<20;}
    void dma_write_to_host(uint64_t a,size_t n,const std::vector<uint8_t>& v,DmaCallback cb,DmaRequestKind) override {
        wr_to_globalmem(a,n,v);cb(true);
    }
    void dma_write_from_globalmem_to_host(uint64_t a,uint64_t b,size_t n,DmaCallback cb,DmaRequestKind) override {
        std::vector<uint8_t> v;rd_from_globalmem(a,n,v);wr_to_globalmem(b,n,v);cb(true);
    }
    void dma_read_from_host_to_globalmem(uint64_t a,size_t n,uint64_t b,DmaCallback cb,DmaRequestKind,const DmaConsumerMetadata&) override {
        std::vector<uint8_t> v;rd_from_globalmem(a,n,v);wr_to_globalmem(b,n,v);cb(true);
    }
    void dma_update_consumer_progress(uint64_t,uint64_t,const DmaConsumerMetadata&) override{}
    uint64_t dma_write_to_host_async(uint64_t,size_t,const std::vector<uint8_t>&) override{assert(false);return 0;}
    bool dma_completion_done(uint64_t) const override{return true;}
    void dma_completion_retire(uint64_t) override{}
    bool controlNetworkAvailable() const override{return false;}
    bool sendControlMessage(uint32_t,const ControlTransportMessage&) override{return false;}
    void setControlMessageHandler(ControlMessageHandler) override{}
    template<class T>void put(uint64_t a,const T& v) {
        std::vector<uint8_t> raw(sizeof(v));std::memcpy(raw.data(),&v,sizeof(v));wr_to_globalmem(a,raw.size(),raw);
    }
    template<class T>T get(uint64_t a) {
        std::vector<uint8_t> raw;rd_from_globalmem(a,sizeof(T),raw);T v;std::memcpy(&v,raw.data(),sizeof(v));return v;
    }
};

class SFUPrecisionTest : public Component {
public:
    SST_ELI_REGISTER_COMPONENT(SFUPrecisionTest,"golem","SFUPrecisionTest",
        SST_ELI_ELEMENT_VERSION(1,0,0),"SFU precision regression",COMPONENT_CATEGORY_UNCATEGORIZED)
    SST_ELI_DOCUMENT_PARAMS({"output","Numerical evidence file",""})
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS({"sfu","Production SFU","SST::Golem::SFUAPI"})
    SFUPrecisionTest(ComponentId_t id,Params& p):Component(id),out(p.find<std::string>("output")) {
        sfu=loadUserSubComponent<SFUAPI>("sfu");assert(sfu);
        Params mp;gm=static_cast<SFUTestMemory*>(loadAnonymousSubComponent<GlobalMemoryAPI>(
            "golem.SFUTestMemory","gm",0,ComponentInfo::SHARE_NONE,mp));assert(gm);
        sfu->bindGlobalMemory(gm);sfu->setCoreInfo(0,1);
        registerAsPrimaryComponent();primaryComponentDoNotEndSim();
        registerClock("1GHz",new Clock::Handler2<SFUPrecisionTest,&SFUPrecisionTest::tick>(this));
        out<<std::setprecision(17);
    }
    SFUAPI* sfu;SFUTestMemory* gm;std::ofstream out;
    unsigned phase=0;bool pending=false,done=false,inputReady=false;uint64_t now=0;
    SFUVectorResult result;std::vector<double>x,y;std::vector<uint32_t>positions{0};
    void emit(const std::string& name,const std::vector<double>& values) {
        out<<name<<" "<<now;for(double v:values)out<<" "<<v;out<<"\n";
    }
    bool tick(Cycle_t cycle) {
        now=cycle;assert(cycle<100000);
        if(phase<5) {
            if(!pending) {
                x.resize(258);y.resize(258);
                for(size_t i=0;i<x.size();++i) {
                    x[i]=phase==4 ? .0137*(1+(i%13)*.113) : phase==3 ? -1.137+(i%97)*.0217 : -12+(i%97)*.25;
                    y[i]=.25+(i%13)*.125;
                }
                SFUVectorRequest r;r.input=&x;r.coefficients=&y;r.rowWidth=x.size();
                const SFUVectorOp ops[]={SFUVectorOp::Add,SFUVectorOp::Multiply,SFUVectorOp::Silu,SFUVectorOp::Rope,SFUVectorOp::RmsNorm};
                r.op=ops[phase];r.positions=&positions;r.activeWidth=x.size();r.coefficientStride=x.size();
                assert(sfu->issueVectorOp(r,&result));pending=true;
            }
            if(result.completed && getCurrentSimCycle()>=result.readyTick) {
                const char*names[]={"add","multiply","silu","rope","rmsnorm"};
                emit(names[phase],result.values);pending=false;++phase;
            }
            return false;
        }
        if(phase<9) {
            const unsigned tile=(phase-5)%2;
            const bool direct=phase>=7;
            AttentionClusterTag tag;tag.generation=1;tag.jobId=direct?100:99;tag.kvTileIndex=tile;
            if(!pending) {
                std::vector<float> scores;
                for(unsigned i=0;i<35;++i) {
                    const float score=float(-2+(i%17)*.217+tile*.31);
                    gm->put<float>(0x2000+i*4,score);scores.push_back(score);
                }
                pending=true;done=false;inputReady=!direct;
                if(direct) {
                    assert(sfu->reserveAttentionScoreSlot(0,tag,35));
                    assert(sfu->reserveAttentionPSlot(0,tag,35));
                    assert(sfu->writeAttentionScoreBeatAsync(0,tag,0,scores,777,
                        [this](bool ok,uint64_t){assert(ok);inputReady=true;}));
                }
            }
            if(inputReady) {
                inputReady=false;
                AttentionTileRequest r;r.jobId=tag.jobId;r.tag=100+phase;r.rows=1;r.cols=35;r.headDim=64;
                r.localScoreAddr=0x2000;r.numKvTiles=2;r.kvTileIndex=tile;r.keyBegin=tile*35;
                r.firstTileForJob=tile==0;r.causal=false;
                r.generation=1;r.directScoreMode=direct;r.scoreSlot=0;r.scoreTag=tag;
                r.directPMode=direct;r.pSlot=0;r.pTag=tag;
                assert(sfu->issueAttentionTile(r,[this,tile,direct,tag](bool ok,const AttentionTileResult& r){
                    assert(ok);
                    if(direct) {
                        const float scale=r.oldOutputScale[0];
                        assert(sfu->readAttentionPRowAsync(0,tag,0,35,778,
                            [this,tile,tag,scale](bool valid,uint64_t,const std::vector<float>& values){
                                assert(valid);std::vector<double> v{scale};v.insert(v.end(),values.begin(),values.end());
                                emit(tile==0?"direct0":"direct1",v);
                                assert(sfu->releaseAttentionScoreSlot(0,tag));
                                assert(sfu->releaseAttentionPSlot(0,tag));done=true;
                            }));
                    } else {
                        std::vector<double> v{r.oldOutputScale[0]};
                        for(unsigned i=0;i<35;++i)v.push_back(gm->get<float>(0x2000+i*4));
                        emit(tile==0?"attention0":"attention1",v);done=true;
                    }
                }));
            }
            if(done){pending=false;++phase;}
            return false;
        }
        if(phase<12) {
            // Exercise descriptor primitives and legacy row Softmax as well.
            if(!pending) {
                for(unsigned i=0;i<65;++i)gm->put<float>(0x4000+i*4,float(.0137*(1+(i%13)*.113)));
                if(phase<11) {
                    SFUPrimitiveDesc d{};d.input0_gm_addr=0x4000;d.output_gm_addr=0x5000;
                    d.elem_count=65;d.dtype=1;d.input0_stride_bytes=d.output_stride_bytes=4;
                    d.op=uint32_t(phase==9?SFUPrimitiveOp::REDUCE_SUM:SFUPrimitiveOp::SIGMOID);
                    gm->put(0x6000,d);assert(sfu->issuePrimitive(0x6000,phase));
                } else {
                    SFUJobDesc d{};d.job_id=17;d.input0_addr=0x4000;d.output_addr=0x5000;
                    d.op_type=uint32_t(SFUJobOp::SOFTMAX_ROW);d.dtype=1;d.rows=1;d.cols=65;
                    d.worker_cores=1;d.chunk_elems=17;gm->put(0x6000,d);assert(sfu->issueJob(0x6000,phase));
                }
                pending=true;
            }
            uint64_t status=0;
            if(sfu->wait(phase,&status)) {
                assert(status==0);std::vector<double> v;
                for(unsigned i=0;i<(phase==9?1:65);++i)v.push_back(gm->get<float>(0x5000+i*4));
                emit(phase==9?"reduce_sum":phase==10?"sigmoid":"softmax_row",v);pending=false;++phase;
            }
            return false;
        }
        if(phase==12) {
            if(!pending) {
                for(unsigned i=0;i<65;++i) {
                    gm->put<uint16_t>(0x8000+i*2,golem_float_to_fp16(float(.0137*(1+(i%13)*.113))));
                    gm->put<uint16_t>(0x9000+i*2,golem_float_to_fp16(float(.25+(i%13)*.125)));
                }
                SFUJobDesc d{};d.job_id=18;d.input0_addr=0x8000;d.input1_addr=0x9000;
                d.output_addr=0xa000;d.scratch_addr=0xb000;d.rows=1;d.cols=65;
                d.elem_count=d.chunk_elems=65;d.worker_cores=1;d.dtype=2;
                d.op_type=uint32_t(SFUJobOp::RMSNORM);
                const float epsilon=1.e-6f;uint32_t bits;std::memcpy(&bits,&epsilon,sizeof(bits));d.reserved1=bits;
                gm->put(0x6000,d);assert(sfu->issueJob(0x6000,phase));pending=true;
            }
            uint64_t status=0;
            if(sfu->wait(phase,&status)) {
                assert(status==0);std::vector<double> v;
                for(unsigned i=0;i<65;++i)v.push_back(golem_fp16_to_float(gm->get<uint16_t>(0xa000+i*2)));
                emit("rmsnorm_job",v);pending=false;++phase;
            }
            return false;
        }
        if(phase==13) {
            if(!pending) {
                SFUSoftmaxTileDesc d{};d.job_id=19;d.local_input_gm_addr=0x4000;d.local_output_gm_addr=0x5000;
                d.global_m=d.block_m=d.valid_m=1;d.global_n=d.block_n=d.valid_n=65;
                d.n_tiles_per_row=1;d.elem_bytes=4;gm->put(0x6000,d);
                assert(sfu->issueSoftmaxTile(0x6000,phase));pending=true;
            }
            uint64_t status=0;
            if(sfu->wait(phase,&status)) {
                assert(status==0);std::vector<double> v;
                for(unsigned i=0;i<65;++i)v.push_back(gm->get<float>(0x5000+i*4));
                emit("softmax_tile",v);pending=false;++phase;
            }
            return false;
        }
        if(phase==14) {
            if(!pending) {
                x.resize(258);y.resize(258);
                for(size_t i=0;i<x.size();++i) {x[i]=-12+(i%97)*.25;y[i]=.25+(i%13)*.125;}
                SFUVectorRequest r;r.op=SFUVectorOp::SiluMultiply;
                r.input=&x;r.coefficients=&y;r.rowWidth=x.size();
                assert(sfu->issueVectorOp(r,&result));
                assert(!result.completed && result.values.empty());
                pending=true;x.clear();y.clear();
            }
            if(result.completed && getCurrentSimCycle()>=result.readyTick) {
                emit("silu_multiply",result.values);pending=false;++phase;
            }
            return false;
        }
        if(phase>=15 && phase<=18) {
            if(!pending) {
                const size_t width=phase==17?8192:65;
                x.assign(width,phase==16?300.:phase==17?4.:1.e-5);y.assign(width,1.);
                if(phase==15) x={-12.,-11.09,-11.,0.,11.,12.};
                SFUVectorRequest r;r.op=phase==15?SFUVectorOp::Silu:SFUVectorOp::RmsNorm;
                r.input=&x;r.coefficients=&y;r.rowWidth=x.size();
                assert(sfu->issueVectorOp(r,&result));pending=true;
            }
            if(result.completed && getCurrentSimCycle()>=result.readyTick) {
                const char* names[]={"boundary_silu","boundary_square_overflow","boundary_sum_overflow","boundary_tiny_norm"};
                emit(names[phase-15],result.values);pending=false;++phase;
            }
            return false;
        }
        if(phase==19 || phase==20) {
            if(!pending) {
                for(unsigned i=0;i<8192;++i)gm->put<float>(0xc000+i*4,1.f);
                if(phase==19) {
                    SFUPrimitiveDesc d{};d.input0_gm_addr=0xc000;d.output_gm_addr=0x16000;
                    d.elem_count=8192;d.dtype=1;d.input0_stride_bytes=d.output_stride_bytes=4;
                    d.op=uint32_t(SFUPrimitiveOp::REDUCE_SUM);
                    gm->put(0x6000,d);assert(sfu->issuePrimitive(0x6000,phase));
                } else {
                    SFUJobDesc d{};d.job_id=20;d.input0_addr=0xc000;d.output_addr=0x16000;
                    d.op_type=uint32_t(SFUJobOp::SOFTMAX_ROW);d.dtype=1;d.rows=1;d.cols=8192;
                    d.worker_cores=1;d.chunk_elems=64;gm->put(0x6000,d);assert(sfu->issueJob(0x6000,phase));
                }
                pending=true;
            }
            uint64_t status=0;
            if(sfu->wait(phase,&status)) {
                assert(status==0);std::vector<double> v;
                for(unsigned i=0;i<(phase==19?1:8192);++i)v.push_back(gm->get<float>(0x16000+i*4));
                emit(phase==19?"boundary_long_reduce":"boundary_long_softmax",v);pending=false;++phase;
            }
            return false;
        }
        out.flush();primaryComponentOKToEndSim();return true;
    }
};
