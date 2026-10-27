#include <sst/core/component.h>
#include <sst/elements/golem/array/computeArray.h>
#include <sst/elements/golem/computeunit/multiPrecisionComputeUnit.h>
#include <sst/elements/golem/computeunit/peTiming.h>
#include <stdexcept>
#include <cstdio>
using namespace SST;
using namespace SST::Golem;
class PackedArrayDriver : public Component {
public:
    SST_ELI_REGISTER_COMPONENT(PackedArrayDriver,"llmalign","PackedArrayDriver",
        SST_ELI_ELEMENT_VERSION(1,0,0),"Packed TU numerical and event timing check",COMPONENT_CATEGORY_UNCATEGORIZED)
    SST_ELI_DOCUMENT_PARAMS({"functional", "Check numerical results when enabled", "1"})
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS({"array","Native compatible array","SST::Golem::ComputeArray"})
    PackedArrayDriver(ComponentId_t id,Params& params):Component(id) {
        functional=params.find<int>("functional",1)!=0;
        array=loadUserSubComponent<ComputeArray>("array",ComponentInfo::SHARE_NONE,
            getTimeConverter("1ns"),new Event::Handler2<PackedArrayDriver,&PackedArrayDriver::done>(this));
        if (!array) throw std::runtime_error("array load failed");
        registerAsPrimaryComponent();primaryComponentDoNotEndSim();
    }
    void setup() override { launch(); }
    void launch() {
        using M=XTileMC::ComputeMode;
        const auto mode=static_cast<M>(code);
        const uint32_t ks[]={1,3,17,64};k=ks[ki];
        XTileMC::PrecisionSpecV02 spec{code,XTileMC::inputBits(mode),XTileMC::integerMode(mode)?0u:1u,32,1,{0,0,0}};
        auto invalid=spec;invalid.arithmetic_version=2;
        if(array->configurePrecision(invalid,k) || array->configurePrecision(spec,0) || array->configurePrecision(spec,65))
            throw std::runtime_error("invalid precision accepted");
        if(!array->configurePrecision(spec,k))throw std::runtime_error("valid precision rejected");
        const uint32_t ones[]={1,1,0x3c00,0x3f80,0x3f800000,0x3f800000,0x38,0x3c};
        for(uint32_t tu=0;tu<2;++tu){
            if(functional) for(uint32_t col=0;col<k;++col){
                array->setVectorBits(tu,col,ones[code]);
                for(uint32_t row=0;row<3;++row)array->setMatrixBits(tu,row*64+col,ones[code]);
            }
            array->configureOutputMode(tu,1);
            for(uint32_t row=0;row<3;++row)array->setOutputBits(tu,row,XTileMC::integerMode(mode)?7:floatToBits(7.f));
        }
        start=getCurrentSimTime("1GHz");pending=2;
        for(uint32_t tu=0;tu<2;++tu)array->beginComputationActive(tu,k);
    }
    void done(Event* event){
        auto* ev=dynamic_cast<ArrayEvent*>(event);
        if(!ev)throw std::runtime_error("event ABI changed");
        const auto mode=static_cast<XTileMC::ComputeMode>(code);
        const auto cycles=peTiming(mode).completionCycles((k+XTileMC::lanesPerBeat(mode)-1)/XTileMC::lanesPerBeat(mode));
        if(getCurrentSimTime("1GHz")-start!=cycles)throw std::runtime_error("PE event cycles differ");
        if(functional)for(uint32_t row=0;row<3;++row){
            const uint32_t expected=XTileMC::integerMode(mode)?k+7:floatToBits(float(k+7));
            if(array->outputBits(ev->getArrayID(),row)!=expected)throw std::runtime_error("packed dot or seed differs");
        }
        delete event;
        if(--pending)return;
        if(++ki==4){ki=0;++code;}
        if(code<8){launch();return;}
        array->useLegacyPrecision();
        if(array->getArrayLatency(0)!=19)throw std::runtime_error("legacy latency not restored");
        std::printf("[PASS] native packed TU: 8 modes x 4 K tails x 2 TUs x 3 CUs, seeds, callbacks, PE cycles, legacy restore; functional=%d\n",functional);
        primaryComponentOKToEndSim();
    }
private:
    ComputeArray* array=nullptr;bool functional=true;
    uint32_t code=0,ki=0,k=0,pending=0;uint64_t start=0;
};
