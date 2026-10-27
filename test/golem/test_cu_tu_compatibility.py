import subprocess
import tempfile
import unittest
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'configure.ac').is_file())
class CuTuCompatibilityTest(unittest.TestCase):
    def test_exact_legacy_reduction_active_k_and_projection_rounding(self):
        body=r"""
#include <sst/elements/golem/tileunit/tileUnit.h>
#include <cassert>
#include <cstring>
#include <random>
using namespace SST::Golem;
int main(){
 std::mt19937 rng(17);
 for(unsigned rows:{1u,16u,64u}){
  TileUnit<float> tu(rows);
  for(unsigned k:{1u,17u,64u})for(unsigned row=0;row<rows;++row){
   float a[64],b[64];
   for(unsigned i=0;i<64;++i){a[i]=(int(rng()%2000)-1000)/64.f;b[i]=(int(rng()%2000)-1000)/64.f;}
   float expected=0;GolemFp16 half;
   for(unsigned i=0;i<k;++i){expected+=a[i]*b[i];half+=GolemFp16(a[i])*GolemFp16(b[i]);}
   float actual=tu.dot(row,a,b,k);
   assert(std::memcmp(&expected,&actual,4)==0);
   assert(tu.projectionDot(row,a,b,k).bits==half.bits);
  }
 }
 float a[]={1e20f,-1e20f,3.f},b[]={1,1,1};TileUnit<float> tu(1);
 assert(tu.dot(0,a,b,3)==3.f);
}
"""
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'test.cpp';source.write_text(body);binary=source.with_suffix('')
            subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(ROOT/'src'),str(source),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)
if __name__=='__main__':unittest.main()
