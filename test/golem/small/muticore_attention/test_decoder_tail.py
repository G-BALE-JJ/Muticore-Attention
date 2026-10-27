"""Run the production DecoderOpJob with deterministic async API test doubles."""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p / 'configure.ac').is_file())
STUB=r'''
#pragma once
#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <map>
#include <vector>
#include <sst/elements/golem/fp16.h>
#include <sst/elements/golem/sfu/vector_math.h>
namespace SST { namespace Golem {
enum class LocalMemoryClient {RoCC};
enum class AttentionClusterTrafficClass {ProjectionWeights,ProjectionInputScatter,ProjectionOutput};
struct SFUVectorResult {std::vector<double> values; uint64_t readyTick=0; bool completed=false;};
struct SFUAPI {bool issueVectorOp(const SFUVectorRequest&,SFUVectorResult*) {return false;}};
struct GlobalMemoryAPI {
 std::map<uint64_t,uint8_t> bytes;
 std::vector<std::pair<uint64_t,size_t>> reads,writes;
 uint64_t getBaseAddr() const {return 0;} uint64_t getSize() const {return 2*1024*1024;}
 size_t localMaxRequestBytes() const {return 4096;}
 bool localReadAsync(uint64_t addr,size_t size,LocalMemoryClient,uint64_t tag,
  std::function<void(bool,uint64_t,const std::vector<uint8_t>&)> cb) {
  std::vector<uint8_t> data(size);for(size_t i=0;i<size;++i)data[i]=bytes[addr+i];cb(true,tag,data);return true;
 }
 bool localWriteAsync(uint64_t addr,const std::vector<uint8_t>& data,LocalMemoryClient,uint64_t tag,
  std::function<void(bool,uint64_t)> cb) {
  for(size_t i=0;i<data.size();++i)bytes[addr+i]=data[i];cb(true,tag);return true;
 }
 void dma_read_from_host_to_globalmem(uint64_t src,size_t size,uint64_t dst,std::function<void(bool)> cb) {
  reads.push_back({src,size});for(size_t i=0;i<size;++i)bytes[dst+i]=bytes[src+i];cb(true);
 }
 void dma_write_from_globalmem_to_host(uint64_t src,uint64_t dst,size_t size,std::function<void(bool)> cb) {
  writes.push_back({dst,size});for(size_t i=0;i<size;++i)bytes[dst+i]=bytes[src+i];cb(true);
 }
 void put(uint64_t addr,double v) {uint16_t h=golem_float_to_fp16(v);std::memcpy(&bytes_dummy,&h,2);bytes[addr]=h&255;bytes[addr+1]=h>>8;}
 double get(uint64_t addr) {return golem_fp16_to_float(uint16_t(bytes[addr])|(uint16_t(bytes[addr+1])<<8));}
 uint16_t bytes_dummy=0;
};
struct WorkerCommandProcessorAPI {
 bool partialOwned=false;std::vector<uint8_t> partial;uint64_t partialWriteReady=0;
 bool beginProjectionPartialBuffer(size_t bytes) {
  if(partialOwned || bytes>1048576)return false;partialOwned=true;partial.resize(bytes);return true;
 }
 bool writeProjectionPartial(uint64_t offset,const std::vector<uint8_t>& data,uint64_t cycle,uint64_t& ready) {
  assert(partialOwned && offset+data.size()<=partial.size());
  std::copy(data.begin(),data.end(),partial.begin()+offset);ready=cycle+(data.size()+255)/256+31;partialWriteReady=ready;return true;
 }
 bool readProjectionPartial(uint64_t offset,size_t size,uint64_t cycle,std::vector<uint8_t>& data,uint64_t& ready) {
  assert(partialOwned && offset+size<=partial.size());data.assign(partial.begin()+offset,partial.begin()+offset+size);
  ready=std::max(cycle,partialWriteReady)+(size+255)/256+31;return true;
 }
 void endProjectionPartialBuffer() {assert(partialOwned);partialOwned=false;}
 bool writeGemmOutputGroupClassAsync(const std::vector<uint32_t>& a,const std::vector<double>& values,size_t,
  AttentionClusterTrafficClass,uint64_t tag,uint64_t cycle,std::function<void(bool,uint64_t)> cb) {
  assert(values.size()==a.size()*64 && cycle>=partialWriteReady);output=values;cb(true,tag);return true;
 }
 std::vector<double> matrix,input,output;std::vector<size_t> active;
 bool isBusy() {return false;}
 bool programGemmMatrixGroupClassBankAsync(const std::vector<uint32_t>& a,uint32_t,const std::vector<double>& v,size_t,
  AttentionClusterTrafficClass,uint64_t tag,uint64_t,std::function<void(bool,uint64_t)> cb) {
  assert(v.size()==4096);matrix=v;active.push_back(a.size());cb(true,tag);return true;
 }
 bool programGemmInputScatterBankAsync(const std::vector<uint32_t>& a,uint32_t,const std::vector<double>& v,size_t,
  AttentionClusterTrafficClass,uint64_t tag,uint64_t,std::function<void(bool,uint64_t)> cb) {
  assert(v.size()==a.size()*64);input=v;cb(true,tag);return true;
 }
 bool launchGemmArrayGroupActiveBank(const std::vector<uint32_t>& a,uint32_t,uint64_t mode,uint32_t,uint64_t cycle,
  std::function<void(uint32_t,uint64_t)> cb) {
  if (!mode)output.assign(a.size()*64,0);
  for(size_t r=0;r<a.size();++r)for(size_t n=0;n<64;++n) {
   double dot=0;
   for(size_t k=0;k<64;++k)dot=vectorFp16Value(dot+vectorFp16Value(input[r*64+k]*matrix[n*64+k]));
   output[r*64+n]=vectorFp16Value(output[r*64+n]+dot);
  }
  for(auto id:a)cb(id,cycle);return true;
 }
 bool readGemmOutputGroupClassAsync(const std::vector<uint32_t>& a,size_t,AttentionClusterTrafficClass,
  uint64_t tag,uint64_t,std::function<void(bool,uint64_t,const std::vector<double>&)> cb) {
  assert(output.size()==a.size()*64);cb(true,tag,output);return true;
 }
};
}}
'''
SOURCE=r'''
#include <sst/elements/golem/attention/decoderOpJob.h>
using namespace SST::Golem;
void run(uint32_t rows,uint32_t lane,uint32_t manager,uint32_t tileBegin=0) {
 GlobalMemoryAPI gm;WorkerCommandProcessorAPI wcp;SFUAPI sfu;DecoderOpJob job(&gm,&wcp,&sfu);
 DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);d.op=1;d.rows=rows;
 d.input_dim=128;d.output_dim=64;d.lane=lane;d.manager=manager;d.output_tile_begin=tileBegin;
 d.input_addr=0x10000000;d.output_addr=0x20000000;d.node_stride=0x08000000;d.weights_offset=0x01000000;
 for(uint32_t r=0;r<rows;++r)for(uint32_t k=0;k<128;++k)gm.put(d.input_addr+(r*128+k)*2,r*.125+k*.0078125);
 for(uint32_t k=0;k<2;++k)for(uint32_t n=0;n<64;++n)for(uint32_t j=0;j<64;++j)
  gm.put((1+tileBegin%4)*d.node_stride+d.weights_offset+(tileBegin/4*2+k)*8192+(n*64+j)*2,n==j?(k?.5:1):0);
 assert(job.start(d));for(uint64_t t=1;t<100000 && !job.complete();++t)job.tick(t,t*1000);
 assert(job.complete() && job.status()==0);
 uint32_t wantedRows=0;
 for(uint32_t r=lane*64;r<rows;r+=256)wantedRows+=std::min(64u,rows-r);
 const bool active=rows!=1 || (manager==0 && lane==0);
 if (rows==1)wantedRows=active?1:0;
 assert(gm.writes.size()==wantedRows);
 size_t inputBytes=0;
 for(auto e:gm.reads)if(d.input_addr<=e.first && e.first<d.input_addr+rows*256)inputBytes+=e.second;
 assert(inputBytes==wantedRows*256);
 for(auto e:gm.writes) {
  assert(e.second==128);uint32_t r=(e.first-d.output_addr)/128;
  assert(r<rows);
  for(uint32_t n=0;n<64;++n)assert(gm.get(e.first+n*2)==vectorFp16Value(gm.get(d.input_addr+(r*128+n)*2)+vectorFp16Value(.5*gm.get(d.input_addr+(r*128+64+n)*2))));
 }
 if(rows==1)for(auto count:wcp.active)assert(count==1);
}
void embedding(uint32_t rows,uint32_t lane,bool invalid=false) {
 GlobalMemoryAPI gm;WorkerCommandProcessorAPI wcp;SFUAPI sfu;DecoderOpJob job(&gm,&wcp,&sfu);
 DecoderOpDesc d{};d.magic=GOLEM_DECODER_OP_MAGIC;d.size_bytes=sizeof(d);d.op=6;d.rows=rows;
 d.input_dim=9;d.output_dim=64;d.lane=lane;d.manager=0;
 d.input_addr=0x10000000;d.output_addr=0x20000000;d.node_stride=0x08000000;d.weights_offset=0x01000000;
 d.completion_addr=0x30000000;d.epoch=1;
 for(uint32_t r=0;r<rows;++r) {
  uint32_t token=r%9;if(invalid && r==lane*64)token=9;
  for(int b=0;b<4;++b)gm.bytes[d.input_addr+r*4+b]=(token>>(8*b))&255;
 }
 for(uint32_t token=0;token<9;++token)for(uint32_t k=0;k<64;++k)
  gm.put((1+token%4)*d.node_stride+d.weights_offset+(token/4)*128+k*2,token+k*.125);
 assert(job.start(d));for(uint64_t t=1;t<100000 && !job.complete();++t)job.tick(t,t*1000);
 assert(job.complete());assert(wcp.active.empty());
 if(invalid) {assert(job.status()==1);assert(gm.reads.size()==1);assert(gm.writes.empty());return;}
 assert(job.status()==0);size_t wantedRows=0;
 for(uint32_t group=lane*64;group<rows;group+=256)for(uint32_t r=group;r<std::min(group+64,rows);++r) {
  uint32_t token=r%9;++wantedRows;
  for(uint32_t k=0;k<64;++k)assert(gm.get(d.output_addr+r*128+k*2)==token+k*.125);
 }
 assert(gm.writes.size()==wantedRows+1);assert(gm.writes.back().first==d.completion_addr);
 size_t weights=0,tokens=0;
 for(auto e:gm.reads) {
  if(d.input_addr<=e.first && e.first<d.input_addr+rows*4)tokens+=e.second;
  else {assert(e.second==128);++weights;}
 }
 assert(tokens==wantedRows*4);assert(weights==wantedRows);
}
int main() {
 run(1,0,0,128);run(1,1,0,128);run(1,0,1,128);
 run(2,0,0);run(65,0,0);run(65,1,0);run(65,2,0);
 run(257,0,0);run(257,3,0);
 embedding(1,0);embedding(1,1);embedding(65,0);embedding(65,1);embedding(257,0);embedding(256,3);
 embedding(65,1,true);
}
'''


class DecoderTailTest(unittest.TestCase):
    def test_m1_and_partial_row_groups_use_real_job_without_padded_dma(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('globalmemory/globalmemory.h','workercmdproc/workercmdproc.h'):
                path=root/'sst/elements/golem'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('#include "stub.h"\n')
            (root/'stub.h').write_text(STUB);(root/'test.cc').write_text(SOURCE)
            subprocess.run(['g++','-std=c++17','-O2','-I',str(root),'-I',str(ROOT/'src'),str(root/'test.cc'),'-o',str(root/'test')],check=True)
            subprocess.run([str(root/'test')],check=True,capture_output=True)


if __name__=='__main__':unittest.main()
