"""Execute production guest scheduling with only the RoCC submission stubbed."""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
GUEST = ROOT / 'src/sst/elements/golem/tests/small/muticore_attention'


class DecoderGuestScheduleTest(unittest.TestCase):
    def execute(self, body):
        header = (GUEST / 'golem_decoder_runtime.h').read_text()
        start = header.index('static bool submit_decoder_op(')
        end = header.index('static bool run_final_norm(', start)
        header = header[:start] + '''
static bool submit_decoder_op(uint32_t, const SST::Golem::DecoderOpDesc& d) {
    submitted.push_back(d);return true;
}
''' + header[end:]
        source = r'''
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <sst/elements/golem/attention/decoderOpAbi.h>
std::vector<SST::Golem::DecoderOpDesc> submitted;
std::map<std::string,std::string> environment;
unsigned lookups=0;
namespace std {
char* guest_test_getenv(const char* key) {
    ++lookups;auto it=environment.find(key);
    return it==environment.end()?nullptr:it->second.data();
}
}
#define GOLEM_DECODER_STAGE 4
#define getenv guest_test_getenv
''' + header + '\nint main(){\n' + body + '\n}\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'test.cc').write_text(source)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(ROOT / 'src'),
                            '-I', str(GUEST), str(root / 'test.cc'), '-o', str(root / 'test')], check=True)
            subprocess.run([str(root / 'test')], check=True)

    def test_logical_sequence_is_read_once_and_keeps_all_lm_descriptors(self):
        self.execute(r'''
using namespace SST::Golem;
auto layout=makeDecoderMemoryLayout(1024,4,1,64,2,true,128256,
    DecoderMemoryLayout::node_stride,true,false);assert(layout.valid);
environment["GOLEM_DECODER_LOGICAL_SEQUENCE"]="777";
assert(run_lm_head(4,256,layout));assert(lookups==1);assert(submitted.size()==16);
for(unsigned i=0;i<submitted.size();++i){
    const auto& d=submitted[i];
    assert(d.input_addr==4*layout.hbm_node_stride+layout.final_output+8*256*2);
    assert(d.output_tile_begin==i*128 && d.epoch==4+i);
    assert(d.output_dim==(i==15?5376u:8192u));
    assert(bool(d.completion_addr)==(i==15));
}
submitted.clear();lookups=0;environment["GOLEM_DECODER_LOGICAL_SEQUENCE"]="0";
assert(!run_lm_head(4,256,layout));assert(submitted.empty() && lookups==1);
environment["GOLEM_DECODER_LOGICAL_SEQUENCE"]="1025";
assert(!run_lm_head(4,256,layout));assert(submitted.empty());
''')

    def test_snapshot_retains_fused_and_unfused_descriptor_dependencies(self):
        self.execute(r'''
using namespace SST::Golem;
auto layout=makeDecoderMemoryLayout(1024,4,1,64,2,true,128256,
    DecoderMemoryLayout::node_stride,true,false);assert(layout.valid);
for(const char* key:{"GOLEM_DECODER_DOWN_RESIDUAL_FUSION","GOLEM_DECODER_ADD_NORM_FUSION",
                    "GOLEM_DECODER_SILU_MULTIPLY_FUSION","GOLEM_DECODER_FFN_PANEL_FUSION",
                    "GOLEM_DECODER_FFN_DOWN_FUSION"})environment[key]="1";
const auto fused=readDecoderGuestOptions();assert(lookups==5);environment.clear();
for(unsigned layer=0;layer<2;++layer)
    assert(run_decoder(4,256,256,64,layout.o,&layout,layer,fused));
assert(lookups==5 && submitted.size()==6);
for(unsigned layer=0;layer<2;++layer){
    const auto& add=submitted[layer*3+1];const auto& down=submitted[layer*3+2];
    assert(submitted[layer*3].stage==1 && add.stage==2 && down.stage==4);
    assert(add.weights_offset==layout.layers[layer].gamma2 && add.head_dim==layout.norm2);
    assert(!add.completion_addr && down.completion_addr);
    assert(down.residual_offset==layout.h1 && down.epoch==layer+1);
}
submitted.clear();const auto plain=readDecoderGuestOptions();const unsigned count=lookups;
assert(run_decoder(4,256,256,64,layout.o,&layout,0,plain));assert(lookups==count);
const unsigned stages[]={1,2,3,4,5,6,7,8,9};assert(submitted.size()==9);
for(unsigned i=0;i<9;++i)assert(submitted[i].stage==stages[i]);
assert(!submitted[1].head_dim && !submitted[1].weights_offset);
assert(submitted.back().completion_addr);
''')


if __name__ == '__main__':
    unittest.main()
