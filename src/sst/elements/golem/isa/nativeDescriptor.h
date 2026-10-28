#ifndef GOLEM_NATIVE_DESCRIPTOR_H
#define GOLEM_NATIVE_DESCRIPTOR_H
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <utility>
#include <initializer_list>
#include <sst/elements/golem/isa/precision.h>
namespace GolemNative {
constexpr uint32_t Magic=0x4e444d47, Version=1, Gemm=1;
constexpr uint32_t MaxCommandBytes=32768, RegionBytes=1024*1024;
constexpr uint64_t VirtualAddress=0x50000000;
constexpr uint32_t BundleMagic=0x4e424d47;
enum class Status : uint32_t { Success=0, InvalidDescriptor=1, Unsupported=2,
    QueueFull=3, BadToken=4, ExecutionFault=5 };
// Version 1 is independent of legacy WorkerTaskListHeader and donor TileBin ABI.
// Offsets address uint32 node maps inside this command. All fields little-endian.
struct alignas(8) Command {
    uint32_t magic,version,kernel_kind,total_bytes,checksum,flags;
    uint32_t physical_worker,logical_worker,worker_count,precision,m,n,k,reuse;
    uint64_t node_bytes,a_offset,b_offset,c_offset,start_cycle;
    uint32_t data_map,data_count,task_map,task_count,a_map,a_count,b_map,b_count;
};
static_assert(sizeof(Command)==128,"native command ABI");
static_assert(offsetof(Command,start_cycle)==88,"native command relocation");
struct Bundle { uint32_t magic,version,total_bytes,commands,checksum,reserved;
    uint32_t offsets[16],sizes[16]; };
static_assert(sizeof(Bundle)==152,"native bundle ABI");
inline uint32_t checksum(const void* bytes,size_t count) {
    auto p=static_cast<const uint8_t*>(bytes);uint32_t hash=2166136261u;
    for(size_t i=0;i<count;++i) { hash^=(i>=16 && i<20)?0:p[i];hash*=16777619u; }
    return hash;
}
inline Status validate(const Command& c,size_t size) {
    if(c.magic!=Magic || c.total_bytes!=size || size<sizeof(c) ||
       size>MaxCommandBytes || size%8 || (c.flags&~1u) ||
       c.physical_worker>=16 || !c.worker_count || c.worker_count>16 ||
       c.logical_worker>=c.worker_count || !c.m || !c.n || !c.k ||
       c.m%64 || c.n%64 || c.k%64 || (c.reuse!=1 && c.reuse!=2) ||
       c.node_bytes!=128ull*1024*1024 || c.a_offset>=c.b_offset || c.b_offset>=c.c_offset ||
       c.c_offset>=c.node_bytes) return Status::InvalidDescriptor;
    if(c.version!=Version || c.kernel_kind!=Gemm || c.precision>7) return Status::Unsupported;
    const uint64_t tasks=((uint64_t(c.m)/64+c.reuse-1)/c.reuse)*
        ((uint64_t(c.n)/64+c.reuse-1)/c.reuse);
    if(c.data_count==0 || c.data_count>4 || c.task_count!=tasks ||
       c.a_count!=c.m/64 || c.b_count!=c.n/64) return Status::InvalidDescriptor;
    uint32_t next=sizeof(c);
    for(auto pair:{std::pair<uint32_t,uint32_t>{c.data_map,c.data_count},
                  {c.task_map,c.task_count},{c.a_map,c.a_count},{c.b_map,c.b_count}}) {
        if(pair.first!=next || uint64_t(pair.first)+uint64_t(pair.second)*4>size)
            return Status::InvalidDescriptor;
        next+=pair.second*4;
    }
    if((uint64_t(next)+7)/8*8!=size) return Status::InvalidDescriptor;
    return Status::Success;
}
}
#endif
