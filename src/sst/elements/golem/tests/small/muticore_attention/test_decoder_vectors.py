"""Exercise the C++ functional SFU additions independently of SST's scheduler."""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[7]

class DecoderVectorTest(unittest.TestCase):
    def test_cpp_vector_math_and_descriptor_contract(self):
        source=r'''
#include <sst/elements/golem/sfu/vector_math.h>
#include <sst/elements/golem/attention/decoderOpAbi.h>
#include <cassert>
#include <cmath>
using namespace SST::Golem;
int main() {
    static_assert(sizeof(DecoderOpDesc)==96,"guest ABI");
    std::vector<double> x={-2,-1,0,1,2,3,4,5}, y={2,3,4,5,6,7,8,9}, output;
    SFUVectorRequest r{};r.input=&x;r.coefficients=&y;r.rowWidth=4;
    r.op=SFUVectorOp::Add;assert(evaluateSFUVectorOp(r,&output));
    for(size_t i=0;i<x.size();++i)assert(output[i]==x[i]+y[i]);
    r.op=SFUVectorOp::Multiply;assert(evaluateSFUVectorOp(r,&output));
    for(size_t i=0;i<x.size();++i)assert(output[i]==x[i]*y[i]);
    r.op=SFUVectorOp::Silu;assert(evaluateSFUVectorOp(r,&output));
    assert(output[2]==0);assert(output[0]<0);assert(output[3]>0 && output[3]<1);
    for(size_t i=0;i<x.size();++i)assert(std::abs(output[i]-x[i]/(1+std::exp(-x[i])))<0.002);
    std::vector<double> bad={1};r.coefficients=&bad;r.op=SFUVectorOp::Multiply;
    assert(!evaluateSFUVectorOp(r,&output));
    r.coefficients=&y;r.rowWidth=3;assert(!evaluateSFUVectorOp(r,&output));
    r.rowWidth=4;r.op=SFUVectorOp::RmsNorm;r.epsilon=-1;
    assert(!evaluateSFUVectorOp(r,&output));
}
'''
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'test.cc').write_text(source)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(ROOT/'src'),str(p/'test.cc'),'-o',str(p/'test')],check=True)
            subprocess.run([str(p/'test')],check=True)

if __name__=='__main__':unittest.main()
