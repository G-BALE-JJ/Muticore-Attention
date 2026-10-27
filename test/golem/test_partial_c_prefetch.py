from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'configure.ac').is_file())


class PartialCPrefetchTest(unittest.TestCase):
    def test_every_reuse_completion_order_requires_current_tile_zero(self):
        # Drive the production readiness predicate with every completion order.
        # In particular {0,1,1,1}, current=0 used to read a stale tile-zero C.
        body = r'''
#include <sst/elements/golem/workercmdproc/partialCPrefetch.h>
#include <algorithm>
#include <cassert>
using SST::Golem::firstPartialReadyForNextWindow;
int main() {
  assert(!firstPartialReadyForNextWindow({},0));
  assert(!firstPartialReadyForNextWindow({0,1,1,1},0));
  std::vector<unsigned> order{0,1,2,3};
  do {
    std::vector<uint8_t> saved(4,0);
    for (auto current:order) {
      bool expected=saved[0];
      for (unsigned i=0;i<4;++i) if(i!=current && !saved[i]) expected=false;
      assert(firstPartialReadyForNextWindow(saved,current)==expected);
      saved[current]=1;
    }
    assert(firstPartialReadyForNextWindow(saved,order.back()));
  } while(std::next_permutation(order.begin(),order.end()));
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'check.cpp';source.write_text(body)
            binary=Path(directory)/'check'
            subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',
                            '-I'+str(ROOT/'src'),str(source),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)


if __name__ == '__main__':
    unittest.main()
