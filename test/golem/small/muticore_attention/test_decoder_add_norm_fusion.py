"""Exercise the production fused path with the existing finite port fixture."""
from pathlib import Path
from unittest.mock import patch
import unittest
import test_decoder_stage_chain as fixture


class AddNormFusionTest(unittest.TestCase):
    def test_rounded_forwarding_partial_chunks_backpressure_and_chain_ownership(self):
        self.exercise(1088,8)

    def test_formal_width_nonuniform_values_and_partial_last_chunk(self):
        self.exercise(2048,9)

    def exercise(self, hidden, rows):
        original = Path.write_text

        def write(path, data, *args, **kwargs):
            if path.name == 'test.cc':
                data = data.replace('bool residualInput=true){', 'bool residualInput=true,bool fusion=false){')
                data = data.replace(' setenv("GOLEM_DECODER_ELIDE_INTERMEDIATE_OUTPUT",elide?',
                                    ' setenv("GOLEM_DECODER_ADD_NORM_FUSION",fusion?"1":"0",1);\n setenv("GOLEM_DECODER_GAMMA_REUSE","1",1);\n setenv("GOLEM_DECODER_ELIDE_INTERMEDIATE_OUTPUT",elide?')
                data = data.replace('if(retained){d.epoch=8;assert(!job.start(d));d.epoch=7;}', '''
 if(fusion){
  assert(!job.start(d)); // No gamma or Norm destination supplied.
  d.weights_offset=gamma-stride;d.head_dim=norm-stride;
 }
 if(retained){d.epoch=8;assert(!job.start(d));d.epoch=7;}''')
                data = data.replace('d.op=3;d.stage=3;d.input_addr=h1;d.input1_addr=gamma;d.output_addr=norm;execute(d);', '''
 d.op=3;d.stage=3;d.input_addr=h1;d.input1_addr=gamma;d.output_addr=norm;
 if(fusion)assert(!job.start(d));else execute(d);
 std::vector<uint8_t> intermediate;
 for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<h;++c)for(uint32_t b=0;b<2;++b){
  intermediate.push_back(gm.bytes[(retained?DecoderMemoryLayout::decoder_norm_handoff_gm:h1)+(r*h+c)*2+b]);
  const uint64_t packed=((c/64)*rows+r)*128+(c%64)*2+b;
  intermediate.push_back(gm.bytes[retained?DecoderMemoryLayout::decoder_input_gm+packed:norm+(r*h+c)*2+b]);
 }''')
                data = data.replace(' return result;\n}\nint main(){\n const auto wide=',
                                    ' result.insert(result.end(),intermediate.begin(),intermediate.end());\n return result;\n}\nint main(){\n const auto wide=')
                data = data.replace('assert(wide==wideChain(true));', '''
 assert(wide==wideChain(true));
 assert(wide==wideChain(true,false,8,true,true));
 assert(wide==wideChain(true,true,8,true,true));
 assert(wide==wideChain(true,true,7,true,true));
 assert(wide==wideChain(true,true,8,false,true));''')
                begin=data.index('std::vector<uint8_t> wideChain(');end=data.index('int main(){',begin)
                wide=data[begin:end].replace('h=1088,rows=8',f'h={hidden},rows={rows}')
                wide=wide.replace('n<17','n<h/64').replace('n/4*17+n','n/4*(h/64)+n')
                value='(.0137*(1+(r%7)*.17+(c%13)*.113))'
                wide=wide.replace('gm.put(in+(r*h+c)*2,.125)',f'gm.put(in+(r*h+c)*2,{value})')
                wide=wide.replace('gm.put(DecoderMemoryLayout::decoder_norm_handoff_gm+(r*h+c)*2,.125)',
                                  f'gm.put(DecoderMemoryLayout::decoder_norm_handoff_gm+(r*h+c)*2,{value})')
                wide=wide.replace('uint16_t bits=golem_float_to_fp16(.0625);','')
                wide=wide.replace('uint64_t a=DecoderMemoryLayout::decoder_wo_handoff_gm+(r*h+c)*2;',
                                  f'uint64_t a=DecoderMemoryLayout::decoder_wo_handoff_gm+(r*h+c)*2;\n uint16_t bits=golem_float_to_fp16(vectorFp16Value({value})*.5);')
                data=data[:begin]+wide+data[end:]
            return original(path, data, *args, **kwargs)

        with patch.object(Path, 'write_text', write):
            fixture.DecoderStageChainTest().test_bits_traffic_lifetimes_and_shape_fallback()


if __name__ == '__main__':
    unittest.main()
