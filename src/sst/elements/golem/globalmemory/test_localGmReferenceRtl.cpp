#include "Vgolem_gm_bank_cluster.h"
#include "verilated.h"
#include "localGmBankScheduler.h"
#include <array>
#include <cassert>
#include <deque>
#include <iostream>
#include <random>
using SST::Golem::LocalGmBankScheduler;
struct Op {uint64_t id, addr; unsigned start,len; bool write; std::array<uint8_t,128> data;};
int main(int argc,char**argv){
 Verilated::commandArgs(argc,argv); Vgolem_gm_bank_cluster d;
 LocalGmBankScheduler s({8,128,1,32,256}); std::deque<Op> q[2];
 std::vector<uint8_t> mem(2097152); std::mt19937 rng(20261016);
 std::array<std::array<uint8_t,128>,2> response{}; bool valid[2]={false,false}; uint64_t id=0, grants=0,reads=0;
 d.clk=0; d.rst_n=0; d.req_valid_i=0; d.eval(); d.clk=1;d.eval();d.clk=0;d.rst_n=1;d.eval();
 for(unsigned cyc=1;cyc<10000;++cyc){
  for(unsigned lane=0;lane<2;++lane) if(cyc<33 ? lane==0 : (cyc<9900 && (rng()%3))) {
   Op o{};o.id=++id;unsigned index=cyc<33?cyc-1:rng()%32;
   // Exercise low rows and the final 16 lines of the preserved 2 MiB capacity.
   o.addr=uint64_t(index<16?index:16384-32+index)*128; o.write=cyc<33 || rng()%2;
   o.start=cyc<33?0:rng()%128; o.len=cyc<33?128:1+rng()%(128-o.start);
   for(auto&b:o.data)b=rng();q[lane].push_back(o);
   assert(s.enqueue(lane,{o.id,o.addr+o.start,o.len,cyc}));
  }
  d.req_valid_i=0;d.req_write_i=0;
  for(unsigned lane=0;lane<2;++lane){
   for(unsigned w=0;w<32;++w)d.req_wdata_i[lane*32+w]=0;
   for(unsigned w=0;w<4;++w)d.req_wbe_i[lane*4+w]=0;
   d.req_addr_i[2*lane]=0; d.req_addr_i[2*lane+1]=0;
   if(q[lane].empty())continue;auto&o=q[lane].front();d.req_valid_i|=1<<lane;d.req_write_i|=unsigned(o.write)<<lane;
   d.req_addr_i[2*lane]=o.addr+o.start;
   for(unsigned j=o.start;j<o.start+o.len;++j){d.req_wbe_i[lane*4+j/32]|=1u<<(j%32);d.req_wdata_i[lane*32+j/4]|=unsigned(o.data[j])<<(8*(j%4));}
  }
  d.eval();auto gs=s.step(cyc);unsigned mask=0;for(auto&g:gs)mask|=1<<g.lane;
  if(mask!=d.req_ready_o){std::cerr<<"grant mismatch cycle="<<cyc<<" c++="<<mask<<" rtl="<<unsigned(d.req_ready_o)<<"\n";return 1;}
  d.clk=1;d.eval();
  for(unsigned lane=0;lane<2;++lane){assert(bool(d.rsp_valid_o&(1<<lane))==valid[lane]);if(valid[lane])for(unsigned j=0;j<128;++j)assert(uint8_t(d.rsp_rdata_o[lane*32+j/4]>>(8*(j%4)))==response[lane][j]);valid[lane]=false;}
  for(auto&g:gs){++grants;auto&o=q[g.lane].front();assert(g.last&&g.id==o.id);
   if(o.write)for(unsigned j=o.start;j<o.start+o.len;++j)mem[o.addr+j]=o.data[j];
   else {++reads;valid[g.lane]=true;for(unsigned j=0;j<128;++j)response[g.lane][j]=mem[o.addr+j];}
   q[g.lane].pop_front();}
  d.clk=0;d.eval();
 }
 assert(q[0].empty()&&q[1].empty());std::cout<<"PASS reference RTL differential cycles=9999 grants="<<grants<<" reads="<<reads<<" conflicts="<<s.stats().bankConflictCycles<<"\n";
}
