"""Software GM residency across Wo/Add/Norm/FFN with finite timed resources."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_decoder_groups import timed_stub
from test_decoder_down_fusion import SOURCE
from test_decoder_tail import ROOT


class DecoderStageChainTest(unittest.TestCase):
    def test_bits_traffic_lifetimes_and_shape_fallback(self):
        source = SOURCE.replace('bool residual=false) {', '''bool residual=false,uint32_t chain=0,uint32_t inputResidency=0) {
 setenv("GOLEM_ALL_GEMM_REUSE_WINDOW","1",1);
 setenv("GOLEM_DECODER_FFN_LAYOUT","2",1);
 setenv("GOLEM_DECODER_GM_STAGE_CHAIN",chain?"1":"0",1);
 setenv("GOLEM_DECODER_LAYER_RESIDENCY",inputResidency || std::getenv("GOLEM_TEST_LAYER_CHECK")?"1":"0",1);
 setenv("GOLEM_DECODER_INPUT_RESIDENCY",inputResidency?"1":"0",1);''')
        source = source.replace('sfu.step(t*1000);job.tick(t,t*1000);',
                                'sfu.step(t*1000);wcp.step(t);job.tick(t,t*1000);')
        source = source.replace('assert(!wcp.partialOwned);job.retire();sfu.now=0;',
                                'assert(!wcp.partialOwned && wcp.events.empty());job.retire();sfu.now=0;wcp.bufferReady=0;')
        needle = ' DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;'
        source = source.replace(needle, r'''
 constexpr uint64_t wo=0x3d000000,woWeight=0x01800000,gamma=0x3f000000;
 for(uint32_t n=0;n<2;++n)for(uint32_t k=0;k<2;++k)
  for(uint32_t o=0;o<64;++o)for(uint32_t j=0;j<64;++j)
   gm.put((1+n%4)*stride+woWeight+(n/4*2+k)*8192+(o*64+j)*2,
          o==j?(k+1)*.0625:0);
 for(uint32_t i=0;i<128;++i)gm.put(gamma+i*2,.5+(i%3)*.125);
 if(inputResidency){
  const uint64_t residual=stride+residualOffset;
  for(uint32_t r=0;r<std::min(64u,rows-lane*64);++r)for(uint32_t c=0;c<128;++c){
   gm.bytes[DecoderMemoryLayout::decoder_norm_handoff_gm+(uint64_t(r)*128+c)*2]=gm.bytes[residual+(uint64_t(r+lane*64)*128+c)*2];
   gm.bytes[DecoderMemoryLayout::decoder_norm_handoff_gm+(uint64_t(r)*128+c)*2+1]=gm.bytes[residual+(uint64_t(r+lane*64)*128+c)*2+1];
  }
  auto record=SoftwareLayerResidency::encode(6,residual,rows,128,lane,0);
  for(size_t i=0;i<record.size();++i)gm.bytes[SoftwareLayerResidency::record+i]=record[i];
 }
 DecoderOpDesc stage{};stage.magic=GOLEM_DECODER_OP_MAGIC;stage.size_bytes=sizeof(stage);
 stage.op=1;stage.stage=1;stage.rows=rows;stage.lane=lane;stage.input_dim=stage.output_dim=128;
 stage.epoch=7;stage.node_stride=stride;stage.input_addr=input;stage.output_addr=wo;stage.weights_offset=woWeight;
 execute(stage);
 std::vector<uint8_t> stages;
 const auto capture=[&](uint32_t step,uint64_t address){
  for(uint32_t g=lane*64;g<rows;g+=256)for(uint32_t r=g;r<std::min(g+64,rows);++r)
   for(uint32_t c=0;c<128;++c)for(uint32_t b=0;b<2;++b){
    uint64_t a=address+(uint64_t(r)*128+c)*2+b;
    if(chain && rows<=256){
     const uint64_t base=step==1?DecoderMemoryLayout::decoder_wo_handoff_gm:
         step==3?DecoderMemoryLayout::decoder_input_gm:DecoderMemoryLayout::decoder_norm_handoff_gm;
     a=base+(step==3?((c/64)*std::min(64u,rows-lane*64)+(r-lane*64))*128+(c%64)*2:((r-lane*64)*128+c)*2)+b;
    }
    stages.push_back(gm.bytes[a]);
   }
 };
 capture(1,wo);
 stage.op=2;stage.stage=2;stage.input_addr=wo;stage.input1_addr=stride+residualOffset;
 stage.output_addr=stride+residualOffset;stage.weights_offset=0;execute(stage);capture(2,stage.output_addr);
 stage.op=3;stage.stage=3;stage.input_addr=stride+residualOffset;stage.input1_addr=gamma;
 stage.output_addr=input;execute(stage);capture(3,input);
''' + needle, 1)
        begin = source.index(' if(fused) {\n  size_t inputBytes=0,outputBytes=0,weightBytes=0;')
        end = source.index(' return result;', begin)
        source = source[:begin] + r'''
 size_t woRead=0,woWrite=0,normRead=0,normWrite=0,h1Read=0,h1Write=0;
 for(auto e:gm.reads){
  if(wo<=e.first && e.first<wo+uint64_t(rows)*256)woRead+=e.second;
  if(input<=e.first && e.first<input+uint64_t(rows)*256)normRead+=e.second;
  if(stride+residualOffset<=e.first && e.first<stride+residualOffset+uint64_t(rows)*256)h1Read+=e.second;
 }
 for(auto e:gm.writes){
  if(wo<=e.first && e.first<wo+uint64_t(rows)*256)woWrite+=e.second;
  if(input<=e.first && e.first<input+uint64_t(rows)*256)normWrite+=e.second;
  if(stride+residualOffset<=e.first && e.first<stride+residualOffset+uint64_t(rows)*256)h1Write+=e.second;
 }
 const bool hit=chain && rows<=256 && wanted;
 assert(woRead==(hit?0:uint64_t(wanted)*256));assert(woWrite==(hit?0:uint64_t(wanted)*256));
 assert(normWrite==(hit?0:uint64_t(wanted)*256));
 assert(normRead==(hit?1u:2u)*uint64_t(wanted)*256);
 assert(h1Read==(hit?(inputResidency?0u:1u):3u)*uint64_t(wanted)*256);
 assert(h1Write==(hit?0:uint64_t(wanted)*256));
result.insert(result.end(),stages.begin(),stages.end());
''' + source[end:]
        source = source[:source.index('int main()')] + r'''
int main(){
 for(auto shape:std::vector<std::pair<uint32_t,uint32_t>>{{64,0},{33,0},{65,1},{257,0},{65,2}}){
  auto expected=run(shape.first,shape.second,17,16,true,true,true,0);
  assert(run(shape.first,shape.second,17,16,true,true,true,1)==expected);
  assert(run(shape.first,shape.second,17,16,true,true,true,1,1)==expected);
 }
}
'''
        source=source.replace('int main(){',r'''
std::vector<uint8_t> wideChain(bool retained,bool elide=false,uint32_t layers=8,bool residualInput=true){
 setenv("GOLEM_DECODER_ELIDE_INTERMEDIATE_OUTPUT",elide?"1":"0",1);
 setenv("GOLEM_DECODER_LAYERS",std::to_string(layers).c_str(),1);
 setenv("GOLEM_DECODER_LAYER_RESIDENCY",elide || std::getenv("GOLEM_TEST_LAYER_CHECK")?"1":"0",1);
 setenv("GOLEM_DECODER_INPUT_RESIDENCY",elide && residualInput?"1":"0",1);
 setenv("GOLEM_DECODER_GM_STAGE_CHAIN",retained?"1":"0",1);
 setenv("GOLEM_ALL_GEMM_REUSE_WINDOW","1",1);setenv("GOLEM_DECODER_FFN_LAYOUT","2",1);
 GlobalMemoryAPI gm;WorkerCommandProcessorAPI wcp;SFUAPI sfu;DecoderOpJob job(&gm,&wcp,&sfu);
 constexpr uint32_t h=1088,rows=8;
 constexpr uint64_t in=0x40000000,out=0x50000000,stride=0x08000000,weight=0x01800000;
 constexpr uint64_t h1=stride+0x06000000,norm=0x60000000,gamma=0x70000000,final=0x72000000;
 for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<h;++c)gm.put(in+(r*h+c)*2,.125);
 if(elide && residualInput){
  for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<h;++c)
   gm.put(DecoderMemoryLayout::decoder_norm_handoff_gm+(r*h+c)*2,.125);
  const auto record=SoftwareLayerResidency::encode(6,in,rows,h,0,0);
  for(size_t i=0;i<record.size();++i)gm.bytes[SoftwareLayerResidency::record+i]=record[i];
 }
 for(uint32_t c=0;c<h;++c)gm.put(gamma+c*2,1);
 for(uint32_t n=0;n<17;++n)for(uint32_t c=0;c<64;++c)
  gm.put((1+n%4)*stride+weight+(n/4*17+n)*8192+(c*64+c)*2,.5);
 for(uint32_t c=0;c<64;++c){
  gm.put(stride+0x01000000+(c*64+c)*2,.125);
  gm.put(stride+0x01200000+(c*64+c)*2,.25);
  for(uint32_t n=0;n<17;++n)gm.put((1+n%4)*stride+0x01400000+(n/4)*8192+(c*64+c)*2,.5);
 }
 auto execute=[&](const DecoderOpDesc& d){
  assert(job.start(d));
  for(uint64_t t=1;t<1000000 && !job.complete();++t){sfu.step(t*1000);wcp.step(t);job.tick(t,t*1000);if(t%19==0)gm.drain();}
  assert(job.complete() && !job.status() && gm.pending.empty() && gm.delayedReads.empty() && wcp.events.empty() && !wcp.partialOwned);
  job.retire();sfu.now=0;wcp.bufferReady=0;
 };
 DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);d.op=1;d.stage=1;
 d.rows=rows;d.input_dim=d.output_dim=h;d.epoch=7;d.node_stride=stride;d.input_addr=in;d.output_addr=out;d.weights_offset=weight;
 execute(d);
 if(retained){
  uint16_t bits=golem_float_to_fp16(.0625);
  for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<h;++c){
   uint64_t a=DecoderMemoryLayout::decoder_wo_handoff_gm+(r*h+c)*2;
   assert(gm.bytes[a]==uint8_t(bits) && gm.bytes[a+1]==uint8_t(bits>>8));
  }
  assert(gm.writes.empty());
 }
 d.op=2;d.stage=2;d.input_addr=out;d.input1_addr=in;d.output_addr=h1;d.weights_offset=0;
 if(retained){d.epoch=8;assert(!job.start(d));d.epoch=7;}
 execute(d);
 d.op=3;d.stage=3;d.input_addr=h1;d.input1_addr=gamma;d.output_addr=norm;execute(d);
 d.op=9;d.stage=4;d.input_dim=h;d.output_dim=64;d.input_addr=norm;d.input1_addr=0x01200000;
 d.weights_offset=0x01000000;d.down_weights_offset=0x01400000;d.residual_offset=0x06000000;d.output_addr=final;
 d.completion_addr=0x78000000;execute(d);
 std::vector<uint8_t> result;
 const bool omitted=retained && elide && layers>7 && residualInput;
 size_t outputBytes=0;
 for(auto e:gm.writes)if(final<=e.first && e.first<final+rows*h*2)outputBytes+=e.second;
 assert(outputBytes==(omitted?0:rows*h*2));
 for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<h*2;++c)
  result.push_back(gm.bytes[(omitted?DecoderMemoryLayout::decoder_norm_handoff_gm:final)+r*h*2+c]);
 if(std::getenv("GOLEM_TEST_LAYER_CHECK")){
  if(retained){
   const uint64_t base=DecoderMemoryLayout::decoder_norm_handoff_gm;
   for(size_t i=0;i<result.size();++i)assert(gm.bytes[base+i]==result[i]);
  }
  d.op=3;d.stage=10;d.epoch=8;d.input_addr=final;d.input1_addr=gamma;
  d.output_dim=h;d.output_addr=0x76000000;d.residual_offset=0;
  if(retained){
   d.epoch=9;assert(job.start(d));
   for(uint64_t t=1;t<10000 && !job.complete();++t){sfu.step(t*1000);wcp.step(t);job.tick(t,t*1000);if(t%19==0)gm.drain();}
   assert(job.complete() && job.status()!=0 && gm.pending.empty() && gm.delayedReads.empty());
   job.retire();sfu.now=0;d.epoch=8;
  }
  execute(d);
  for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<h*2;++c)result.push_back(gm.bytes[d.output_addr+r*h*2+c]);
 }
 return result;
}
int main(){
 const auto wide=wideChain(false);
 assert(wide==wideChain(true));
 assert(wide==wideChain(true,true));
 assert(wide==wideChain(true,true,7)); // Last layer must still materialize Y.
 assert(wide==wideChain(false,true)); // No GM chain: retain the HBM fallback.
 assert(wide==wideChain(true,true,8,false)); // HBM residual consumer needs the mirror.
 setenv("GOLEM_DECODER_ELIDE_INTERMEDIATE_OUTPUT","0",1);
''')
        stub=timed_stub().replace(
            'for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);return true;',
            'delayedReads.push_back([this,addr,data,tag,cb]{for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);});return true;')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                p=root/'sst/elements/golem'/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(stub);(root/'test.cc').write_text(source)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),str(root/'test.cc'),'-o',str(root/'test')],check=True)
            env=dict(os.environ,GOLEM_REUSE_WINDOW_ARRAY_GROUPS='1',GOLEM_REUSE_WINDOW_PREFETCH_SLOTS='4',
                     GOLEM_REUSE_WINDOW_EARLY_A='1',GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP='1',GOLEM_REUSE_WINDOW_A_PREFETCH='1')
            result=subprocess.run([str(root/'test')],capture_output=True,env=env,timeout=180)
            self.assertEqual(result.returncode,0,result.stderr.decode())

if __name__=='__main__':unittest.main()
