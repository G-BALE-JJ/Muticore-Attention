#include <sst/core/component.h>
#include <sst/elements/golem/tileengine/tileEngine.h>
#include <sst/elements/golem/tileunit/tileUnitAPI.h>
#include <stdexcept>
#include <cstdio>
#include <map>
using namespace SST;
using namespace SST::Golem;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
// Deterministic input transport fixtures. The production TE, TU, CU, buffer
// scheduling, partial-C handling and completion events remain under test.
class FixtureMemory : public GlobalMemoryAPI {
public:
 SST_ELI_REGISTER_SUBCOMPONENT(FixtureMemory,"llmnative","Memory",SST_ELI_ELEMENT_VERSION(1,0,0),"Input/output fixture",GlobalMemoryAPI)
 FixtureMemory(ComponentId_t id,Params& p):GlobalMemoryAPI(id,p){}
 std::map<uint64_t,uint8_t> bytes;
 void wr_to_globalmem(uint64_t a,size_t n,const std::vector<uint8_t>& d) override {require(d.size()>=n,"short write");for(size_t i=0;i<n;++i)bytes[a+i]=d[i];}
 void rd_from_globalmem(uint64_t a,size_t n,std::vector<uint8_t>& d) override {d.resize(n);for(size_t i=0;i<n;++i)d[i]=bytes[a+i];}
 bool localReadAsync(uint64_t a,size_t n,LocalMemoryClient,uint64_t t,LocalReadCallback cb) override {std::vector<uint8_t>d;rd_from_globalmem(a,n,d);cb(true,t,d);return true;}
 bool localWriteAsync(uint64_t a,const std::vector<uint8_t>& d,LocalMemoryClient,uint64_t t,LocalWriteCallback cb) override {wr_to_globalmem(a,d.size(),d);cb(true,t);return true;}
 size_t localMaxRequestBytes() const override{return 4096;}
 void wr_to_network(uint64_t,size_t,std::vector<uint8_t>&) override {throw std::runtime_error("unexpected network write");}
 void rd_to_network(uint64_t,size_t,uint64_t) override {throw std::runtime_error("unexpected network read");}
 void setBaseAddr(uint64_t) override{}
 uint64_t getBaseAddr() const override{return 0;}
 uint64_t getSize() const override{return 1048576;}
 void dma_write_to_host(uint64_t a,size_t n,const std::vector<uint8_t>&d,DmaCallback cb,DmaRequestKind) override{wr_to_globalmem(a,n,d);cb(true);}
 void dma_write_from_globalmem_to_host(uint64_t a,uint64_t b,size_t n,DmaCallback cb,DmaRequestKind) override{std::vector<uint8_t>d;rd_from_globalmem(a,n,d);wr_to_globalmem(b,n,d);cb(true);}
 void dma_read_from_host_to_globalmem(uint64_t a,size_t n,uint64_t b,DmaCallback cb,DmaRequestKind,const DmaConsumerMetadata&) override{std::vector<uint8_t>d;rd_from_globalmem(a,n,d);wr_to_globalmem(b,n,d);cb(true);}
 void dma_update_consumer_progress(uint64_t,uint64_t,const DmaConsumerMetadata&) override{}
 uint64_t dma_write_to_host_async(uint64_t a,size_t n,const std::vector<uint8_t>&d) override{wr_to_globalmem(a,n,d);return 1;}
 bool dma_completion_done(uint64_t) const override{return true;}
 void dma_completion_retire(uint64_t) override{}
 bool controlNetworkAvailable() const override{return false;}
 bool sendControlMessage(uint32_t,const ControlTransportMessage&) override{return false;}
 void setControlMessageHandler(ControlMessageHandler) override{}
};
class FixtureScheduler : public RequestSchedulerAPI {
public:
 SST_ELI_REGISTER_SUBCOMPONENT(FixtureScheduler,"llmnative","Scheduler",SST_ELI_ELEMENT_VERSION(1,0,0),"Panel readiness fixture",RequestSchedulerAPI)
 FixtureScheduler(ComponentId_t id,Params&p):RequestSchedulerAPI(id,p){}
 GlobalMemoryAPI* gm=nullptr;uint64_t serial=0;
 void bindGlobalMemory(GlobalMemoryAPI* g) override{gm=g;}
 uint64_t submitWindowTransaction(const WcpWindowTransaction&t) override {
  require(gm!=nullptr,"unbound fixture");
  const uint32_t kw=std::max(t.kWindowTiles,1u),kt=std::max(t.totalKTileCount,kw);
  for(uint32_t i=0;i<t.kTiles;++i){
   const uint32_t offset=t.useIndependentMatVecTiles?(i/kw)*kt+i%kw:i;
   uint64_t a=t.matBaseAddr+offset*t.matStrideBytes,b=t.vecBaseAddr+offset*t.vecStrideBytes;
#ifdef TEST_PLACEMENT
   if(!t.matSourceAddrs.empty() && i<t.matSourceAddrs.size())a=t.matSourceAddrs[i];
   if(!t.vecSourceAddrs.empty() && i<t.vecSourceAddrs.size())b=t.vecSourceAddrs[i];
#endif
   std::vector<uint8_t>d;
   if(!t.skipMatRead && (!t.useIndependentMatVecTiles || i<t.matTileCount)){
    gm->rd_from_globalmem(a,t.matStrideBytes,d);gm->wr_to_globalmem(t.localMatBaseAddr+(i%t.slotCount)*t.localMatSlotStrideBytes,d.size(),d);
   }
   if(!t.skipVecRead && (!t.useIndependentMatVecTiles || i<t.vecTileCount)){
    gm->rd_from_globalmem(b,t.vecStrideBytes,d);gm->wr_to_globalmem(t.localVecBaseAddr+(i%t.slotCount)*t.localVecSlotStrideBytes,d.size(),d);
   }
  }
  return ++serial;
 }
 bool isTileReady(uint64_t,uint32_t) const override{return true;}
 bool getTileKStepReadiness(uint64_t,uint32_t,uint32_t,bool&a,bool&b) const override{a=b=true;return true;}
 bool getTileTimeline(uint64_t,uint32_t,WcpTileTimelineDebug&) const override{return true;}
 uint64_t getTimelineCycle() const override{return getCurrentSimTime("1GHz");}
 void retireTileReady(uint64_t,uint32_t) override{}
 bool isTransactionDone(uint64_t) const override{return true;}
 void retireTransaction(uint64_t) override{}
};
class NativeEngineDriver : public Component {
public:
 SST_ELI_REGISTER_COMPONENT(NativeEngineDriver,"llmnative","Driver",SST_ELI_ELEMENT_VERSION(1,0,0),"Native TE packed/placement checks",COMPONENT_CATEGORY_UNCATEGORIZED)
 SST_ELI_DOCUMENT_PARAMS({"placed","Use explicit HBM panel mapping","0"},{"replicate","Replicate B panels","0"},{"functional","Check values","1"},{"instance","Independent fixture instance","0"})
 SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS({"array","Production TU","SST::Golem::ComputeArray"},{"engine","Production TE","SST::Golem::WorkerCommandProcessorAPI"},{"memory","Fixture memory","SST::Golem::GlobalMemoryAPI"},{"scheduler","Fixture scheduler","SST::Golem::RequestSchedulerAPI"})
 NativeEngineDriver(ComponentId_t id,Params&p):Component(id){
  instance=p.find<uint32_t>("instance",0);
  const int placementMode=p.find<int>("placed",0);placed=placementMode!=0;packedPlacement=placementMode==2;replicate=p.find<int>("replicate",0);functional=p.find<int>("functional",1);
  a=loadUserSubComponent<TileUnitAPI>("array",ComponentInfo::SHARE_NONE,getTimeConverter("1ns"),new Event::Handler2<NativeEngineDriver,&NativeEngineDriver::arrayDone>(this));
  te=loadUserSubComponent<WorkerCommandProcessorAPI>("engine",ComponentInfo::SHARE_NONE);
  gm=loadUserSubComponent<GlobalMemoryAPI>("memory",ComponentInfo::SHARE_NONE);
  sched=loadUserSubComponent<RequestSchedulerAPI>("scheduler",ComponentInfo::SHARE_NONE);
  require(a&&te&&gm&&sched,"native subcomponent load failed");sched->bindGlobalMemory(gm);
  te->bindResources(instance,nullptr,gm,a,sched,nullptr);
  registerClock("1GHz",new Clock::Handler2<NativeEngineDriver,&NativeEngineDriver::tick>(this));
  registerAsPrimaryComponent();primaryComponentDoNotEndSim();
 }
 void setup() override{if(packedPlacement)code=2;launch();}
 void launch(){
  complete=false;outputs=0;
  WorkerTaskListHeader h;h.worker_slot=0;h.scheduler_worker_slot=instance;h.task_count=1;h.active_worker_cores=1;h.total_groups=1;
  h.data_memory_node_count=placed?4:2;h.mem_node_size=1048576;h.m=placed?6:3;h.n=placed?4:2;h.k=34;
  h.block_m=3;h.block_n=2;h.block_k=17;h.hw_input_size=17;h.hw_output_size=3;h.elem_bytes=4;
  h.local_slot_count=8;h.local_mat_ping_gm_addr=4096;h.local_vec_ping_gm_addr=16384;
  h.local_mat_slot_stride_bytes=256;h.local_vec_slot_stride_bytes=256;
  h.off_gemm_mat_base=0;h.off_gemm_vec_base=65536;h.off_gemm_out_base=131072;
  h.b_reuse_m_tiles=placed?2:1;h.a_reuse_n_tiles=placed?2:1;
  const auto mode=static_cast<XTileMC::ComputeMode>(code);
  XTileMC::PrecisionSpecV02 spec{code,XTileMC::inputBits(mode),XTileMC::integerMode(mode)?0u:1u,32,1,{0,0,0}};
  h.vec_stride_bytes=placed&&!packedPlacement?68:XTileMC::packedRowBytes(17,mode);h.mat_stride_bytes=3*h.vec_stride_bytes;
  const uint32_t ones[]={1,1,0x3c00,0x3f80,0x3f800000,0x3f800000,0x38,0x3c};
  auto payload=[&](unsigned rows,uint32_t raw){
   const unsigned bits=placed&&!packedPlacement?32:XTileMC::inputBits(mode);
   const unsigned rowBytes=unsigned(h.vec_stride_bytes);
   std::vector<uint8_t>data(rows*rowBytes,0);
   for(unsigned row=0;row<rows;++row)for(unsigned k=0;k<17;++k){
    const unsigned bit=k*bits;
    for(unsigned b=0;b<bits;++b)if(raw&(1u<<b))data[row*rowBytes+(bit+b)/8]|=1u<<((bit+b)%8);
   }
   return data;
  };
#ifdef TEST_PLACEMENT
  WorkloadPlacement plan;plan.dataNodes={1+2*instance,2+2*instance};plan.taskNodes={1+2*instance};plan.aNodes={1+2*instance,2+2*instance};plan.bNodes={2+2*instance,1+2*instance};plan.replicateB=replicate;
#endif
  for(uint32_t tile=0;tile<(placed?2u:1u);++tile)for(uint32_t kt=0;kt<2;++kt){
   const float value=float((instance+1)*(tile+1));
   const uint32_t placedBits=packedPlacement?GolemFp16(value).bits:floatToBits(value);
   auto mat=payload(3,placed?placedBits:ones[code]);
   auto vec=payload(2,placed?placedBits:ones[code]);
   uint64_t ma=1048576+h.off_gemm_mat_base+kt*h.mat_stride_bytes;
   uint64_t va=1048576+h.off_gemm_vec_base+kt*2*h.vec_stride_bytes;
#ifdef TEST_PLACEMENT
   if(placed){ma=plan.aAddress(tile,kt,2,h.mem_node_size,h.off_gemm_mat_base,h.mat_stride_bytes);va=plan.bAddress(tile,kt,2,1+2*instance,h.mem_node_size,h.off_gemm_vec_base,2*h.vec_stride_bytes);}
#endif
   gm->wr_to_globalmem(ma,mat.size(),mat);gm->wr_to_globalmem(va,vec.size(),vec);
  }
  require(te->setWindowCallbacks([this,mode](uint64_t task,uint64_t,const std::vector<uint8_t>& tile){
   require(tile.size()==24,"TE output shape differs");
   if(functional)for(unsigned i=0;i<6;++i){uint32_t bits=0;std::memcpy(&bits,tile.data()+4*i,4);
    const float expected=placed?34.f*float((task/2)+1)*float((task%2)+1)*float((instance+1)*(instance+1)):34.f;
    const uint32_t golden=!placed&&XTileMC::integerMode(mode)?34:floatToBits(expected);
    if(bits!=golden)std::fprintf(stderr,"MISMATCH task=%llu element=%u actual=%08x expected=%08x\n",(unsigned long long)task,i,bits,golden);
    require(bits==golden,"TE numerical output differs");}
   ++outputs;return true;
  },[this](uint64_t,uint64_t,uint64_t){complete=true;}),"callbacks rejected");
  bool ok=false;
#ifdef TEST_PLACEMENT
  if(placed)ok=te->startPlacedWindow(h,plan,packedPlacement?&spec:nullptr);
  else
#endif
  ok=te->startPrecisionWindow(h,spec);
  require(ok,"native window rejected");
 }
 bool tick(Cycle_t cycle){
  require(cycle<100000,"native TE timed out");
  if(complete){
   require(outputs==(placed?4u:1u),"TE tile completion count differs");
   if(!placed&&++code<8)launch();
   else{std::printf("[NATIVE_CYCLES] cycles=%llu\n",(unsigned long long)cycle);std::printf("[PASS] native TE: packed=%d placed=%d replicate=%d functional=%d instance=%u\n",!placed,placed,replicate,functional,instance);primaryComponentOKToEndSim();return true;}
  }
  te->tick(cycle);return false;
 }
 void arrayDone(Event* e){auto*ae=dynamic_cast<ArrayEvent*>(e);require(ae!=nullptr,"bad TE array event");te->handleArrayDone(ae->getArrayID(),getCurrentSimTime("1GHz"));delete e;}
private:
 TileUnitAPI*a=nullptr;WorkerCommandProcessorAPI*te=nullptr;GlobalMemoryAPI*gm=nullptr;RequestSchedulerAPI*sched=nullptr;
 bool placed=false,packedPlacement=false,replicate=false,functional=true,complete=false;uint32_t instance=0,code=0,outputs=0;
};
