"""Run the production SFU scheduler with real shared hardware pipeline objects."""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'configure.ac').is_file())

SOURCE = r'''
#include <sst/elements/golem/sfu/vector_pipeline.h>
#include <cassert>
#include <iostream>
#include <set>
using namespace SST::Golem;
using Scheduler = SFUVectorPipeline;
struct Fixture {
    TensorHardwarePipeline vector, exp, reciprocal;
    Fixture(unsigned depth = 16, unsigned ii = 1) {
        vector.configure(3, ii, depth);
        exp.configure(8, ii, depth);
        reciprocal.configure(8, ii, depth);
    }
};
SFUVectorRequest request(SFUVectorOp op, std::vector<double>& x, std::vector<double>& y) {
    SFUVectorRequest r{}; r.op=op; r.input=&x; r.coefficients=&y; r.rowWidth=x.size(); return r;
}
std::vector<double> input(size_t n) {
    std::vector<double> x(n);
    for (size_t i=0;i<n;++i) x[i]=vectorFp16Value(-12.f + (i%97)*.25f);
    return x;
}
std::vector<Scheduler::Completion> run(Scheduler& s, unsigned limit = 200000,
                                     const Scheduler::Trace& trace = {}) {
    std::vector<Scheduler::Completion> all;
    for (unsigned t=0;t<limit && s.active();++t) {
        s.step(t,trace);
        auto done=s.takeCompleted();
        all.insert(all.end(),std::make_move_iterator(done.begin()),std::make_move_iterator(done.end()));
    }
    assert(!s.active()); return all;
}
void numerics(const Scheduler::Completion& c, const std::vector<double>& x,
              const std::vector<double>& y) {
    assert(c.values.size()==x.size());
    for (size_t i=0;i<x.size();++i) {
        const float a=vectorFp16Value(x[i]);
        float z=0;
        if (c.op==SFUVectorOp::Silu) {
            const float e=std::exp(-a), d=1.f+e, inverse=1.f/d;
            z=a*inverse;
            // Independent mathematical reference, including nonzero negative tail.
            assert(std::abs(c.values[i]-a/(1.0+std::exp(-double(a)))) < .008);
        } else z=c.op==SFUVectorOp::Add ? a+vectorFp16Value(y[i]) : a*vectorFp16Value(y[i]);
        assert(c.values[i]==vectorFp16Value(z));
    }
}
void direct() {
    Fixture f;
    Scheduler s(f.vector,f.exp,f.reciprocal,16,4,64,2);
    auto x=input(35); std::vector<double> y(35,2);
    uint64_t id=0;
    assert(s.submit(request(SFUVectorOp::Add,x,y),0,&id));
    for (unsigned t=0;t<5;++t) { s.step(t); assert(s.takeCompleted().empty()); }
    s.step(5);auto result=s.takeCompleted();assert(result.size()==1);
    assert(result[0].completionCycle==5);numerics(result[0],x,y);
    assert(f.vector.acceptedTokens==3 && !f.exp.acceptedTokens && !f.reciprocal.acceptedTokens);
    assert(s.submit(request(SFUVectorOp::Multiply,x,y),6,&id));
    std::vector<Scheduler::Completion> multiplication;
    for(unsigned t=6;s.active();++t) {s.step(t);auto done=s.takeCompleted();
        multiplication.insert(multiplication.end(),done.begin(),done.end());}
    assert(multiplication.size()==1);numerics(multiplication[0],x,y);
}
uint64_t silu(size_t n, bool traceDependencies=false, unsigned expLanes=4) {
    Fixture f;
    Scheduler s(f.vector,f.exp,f.reciprocal,16,expLanes,64,2);
    auto x=input(n);std::vector<double> y;uint64_t id=0;
    assert(s.submit(request(SFUVectorOp::Silu,x,y),0,&id));
    // Asynchronous snapshots must survive the caller clearing operands.
    const auto original=x; x.clear();
    std::vector<Scheduler::Issue> issues;
    auto done=run(s,200000,[&](const auto& issue){issues.push_back(issue);});
    assert(done.size()==1);numerics(done[0],original,y);
    assert(s.fifoHighWater<=64 && ((s.fifoBackpressureEvents>0) == (n>64)));
    assert(f.exp.acceptedTokens==(n+expLanes-1)/expLanes && f.reciprocal.acceptedTokens==n);
    assert(f.vector.acceptedTokens==3*((n+15)/16));
    if(traceDependencies) {
        std::vector<std::array<uint64_t,5>> ready(n);
        for(const auto& issue:issues) {
            assert(issue.job==id);
            for(size_t i=issue.first;i<issue.first+issue.elements;++i) {
                if(issue.stage) assert(issue.issueCycle>=ready[i][issue.stage-1]);
                ready[i][issue.stage]=issue.completionCycle;
            }
        }
        uint64_t lastNeg=0,firstExp=~uint64_t(0);
        for(const auto& i:issues) {
            if(i.stage==Scheduler::Negate)lastNeg=std::max(lastNeg,i.completionCycle);
            if(i.stage==Scheduler::Exp)firstExp=std::min(firstExp,i.issueCycle);
        }
        assert(firstExp<lastNeg); // The next stage starts before all input is processed.
    }
    return done[0].completionCycle;
}
uint64_t contend(bool external, bool reverse=false) {
    Fixture f;
    Scheduler s(f.vector,f.exp,f.reciprocal,16,4,64,3);
    auto x=input(257);std::vector<double> y(257,1.5);uint64_t a=0,b=0;
    assert(s.submit(request(reverse?SFUVectorOp::Add:SFUVectorOp::Silu,x,y),0,&a));
    assert(s.submit(request(reverse?SFUVectorOp::Silu:SFUVectorOp::Add,x,y),0,&b));
    uint64_t end=0;size_t completed=0;std::set<uint64_t> vectorIssueCycles;
    for(unsigned t=0;t<5000 && s.active();++t) {
        // These reservations use the identical objects and API used by the
        // Attention RowEngine. They interleave with already-live SiLU tokens.
        auto other=[&]{if(external && t<200 && t%2==0) {
            uint64_t finish=0;
            f.exp.reserve(t,&finish);
            f.reciprocal.reserve(t,&finish);
        }};
        if(!reverse)other();
        s.step(t,[&](const Scheduler::Issue& issue) {
            if(issue.stage==Scheduler::Negate || issue.stage==Scheduler::AddOne ||
               issue.stage==Scheduler::Multiply || issue.stage==Scheduler::Direct)
                assert(vectorIssueCycles.insert(issue.issueCycle).second);
        });
        if(reverse)other();
        for(const auto& c:s.takeCompleted()) {
            numerics(c,x,y);++completed;
            if(c.op==SFUVectorOp::Silu)end=c.completionCycle;
        }
    }
    assert(!s.active() && completed==2 && s.requestHighWater==2);
    assert(f.vector.acceptedTokens==68 && vectorIssueCycles.size()==68);
    assert(f.vector.maxOccupancy<=16 && f.exp.maxOccupancy<=16 && f.reciprocal.maxOccupancy<=16);
    return end;
}
void backpressure() {
    Fixture f(1,3);
    Scheduler s(f.vector,f.exp,f.reciprocal,16,4,16,2);
    auto x=input(97);std::vector<double> y;uint64_t a=0,b=0,c=0;
    assert(s.submit(request(SFUVectorOp::Silu,x,y),0,&a));
    assert(s.submit(request(SFUVectorOp::Silu,x,y),0,&b));
    assert(!s.submit(request(SFUVectorOp::Silu,x,y),0,&c));
    auto done=run(s);assert(done.size()==2);
    for(const auto& d:done)numerics(d,x,y);
    assert(s.resourceBackpressureEvents>0 && s.fifoBackpressureEvents>0 && s.fifoHighWater<=16);
    assert(f.vector.maxOccupancy==1 && f.exp.maxOccupancy==1 && f.reciprocal.maxOccupancy==1);
    assert(s.submit(request(SFUVectorOp::Silu,x,y),10000,&c)); // Credits returned.
    for(unsigned t=10000;s.active();++t)s.step(t);
    assert(s.takeCompleted().size()==1);
    x.resize(8193);assert(!s.submit(request(SFUVectorOp::Silu,x,y),20000,&c));
}
int main(int argc,char**argv) {
    assert(argc==2);std::string mode=argv[1];
    if(mode=="direct")direct();
    else if(mode=="tail") {silu(1);silu(3);silu(17);silu(35);silu(257,true);}
    else if(mode=="throughput") {
        auto cycles=silu(8192,true);
        assert(cycles>=8192 && cycles<9000); // One scalar reciprocal/cycle is the bottleneck.
        std::cout<<"8192-element SiLU cycles="<<cycles<<"\n";
        auto configured=silu(8192,true,16);
        assert(configured==8216);
        std::cout<<"8192-element SiLU, EXP=16 lanes, cycles="<<configured<<"\n";
    }
    else if(mode=="contention") {
        const auto clean=contend(false),busy=contend(true);
        assert(busy>clean);assert(contend(true,true)>contend(false,true));
        std::cout<<"SiLU shared-resource cycles: clean="<<clean<<" busy="<<busy<<"\n";
    }
    else if(mode=="backpressure")backpressure();
    else assert(false);
}
'''


class SFUVectorPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        p = Path(cls.temp.name)
        (p / 'test.cc').write_text(SOURCE)
        cls.exe = p / 'test'
        subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                        '-I', str(ROOT / 'src'), str(p / 'test.cc'), '-o', str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_add_multiply_and_completion_visibility(self):
        subprocess.run([str(self.exe), 'direct'], check=True)

    def test_tail_numerics_and_element_dependencies(self):
        subprocess.run([str(self.exe), 'tail'], check=True)

    def test_streaming_throughput(self):
        subprocess.run([str(self.exe), 'throughput'], check=True)

    def test_shared_exp_reciprocal_and_vector_contention(self):
        subprocess.run([str(self.exe), 'contention'], check=True)

    def test_finite_queues_backpressure_and_admission(self):
        subprocess.run([str(self.exe), 'backpressure'], check=True)


if __name__ == '__main__':
    unittest.main()
