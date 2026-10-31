"""Check all binary16 rounding boundaries against NumPy's independent converter."""
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'configure.ac').is_file())


class SFUArithmeticTest(unittest.TestCase):
    def test_binary16_midpoints_subnormals_overflow_and_special_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'test.cc').write_text(r'''
#include <sst/elements/golem/sfu/arithmetic.h>
#include <iostream>
#include <cassert>
using namespace SST::Golem;
int main() {
    SFUArithmetic arithmetic(SFUInternalPrecision::FP16);
    assert(SFUArithmetic::parse("fp16")==SFUInternalPrecision::FP16);
    assert(SFUArithmetic::parse("fp32")==SFUInternalPrecision::FP32);
    bool rejected=false;
    try { SFUArithmetic::parse("fp64"); } catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected);
    assert(std::isinf(arithmetic.mul(256.f,256.f)));
    float sum=0;
    for(unsigned i=0;i<4096;++i)sum=arithmetic.add(sum,1.f);
    assert(sum==2048.f);
    assert(SFUArithmetic{}.fp16());
    assert(SFUArithmetic{}.add(2048.f,1.f)==2048.f);
    assert(SFUArithmetic(SFUInternalPrecision::FP32).add(2048.f,1.f)==2049.f);
    double input;
    while(std::cin.read(reinterpret_cast<char*>(&input),sizeof(input))) {
        const double output=arithmetic.round(input);
        std::cout.write(reinterpret_cast<const char*>(&output),sizeof(output));
    }
}
''')
            subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT/'src'),
                            str(path/'test.cc'),'-o',str(path/'test')],check=True)
            finite = np.arange(0x7c00,dtype=np.uint16).view(np.float16).astype(np.float64)
            mids = (finite[:-1]+finite[1:])*.5
            values = np.r_[finite,mids,np.nextafter(mids,np.inf),np.nextafter(mids,-np.inf),
                           65520.,np.nextafter(65520.,0),np.nextafter(65520.,np.inf),
                           np.inf,np.nan]
            values = np.r_[values,-values]
            output = subprocess.run([str(path/'test')],input=values.tobytes(),capture_output=True,check=True)
            actual = np.frombuffer(output.stdout,dtype=np.float64)
            with np.errstate(over='ignore',invalid='ignore'):
                expected = values.astype(np.float16).astype(np.float64)
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(np.signbit(actual[~np.isnan(expected)]),
                                          np.signbit(expected[~np.isnan(expected)]))


if __name__ == '__main__':
    unittest.main()
