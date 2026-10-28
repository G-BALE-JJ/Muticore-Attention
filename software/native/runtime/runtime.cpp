#include <runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
namespace GolemNative {
int64_t submit(uint64_t address,uint32_t bytes) {
    int64_t result;
    asm volatile("fence rw,rw\n.insn r 0x0b, 7, 0x15, %0, %1, %2"
        : "=r"(result):"r"(address),"r"(uint64_t(bytes)):"memory");
    return result;
}
Status wait(Token token) {
    uint64_t result;
    asm volatile(".insn r 0x0b, 7, 0x16, %0, %1, x0\nfence rw,rw"
        : "=r"(result):"r"(token):"memory");
    return static_cast<Status>(result);
}
int runForCore(uint32_t core,const void* memory,size_t capacity) {
    auto error=[core](const char* stage,int code) {std::printf("[NATIVE_GUEST_ERROR] core=%u stage=%s code=%d\n",core,stage,code);return code;};
    const auto* bytes=static_cast<const uint8_t*>(memory);
    Bundle b{};std::memcpy(&b,bytes,sizeof(b));
    if(b.magic!=BundleMagic || b.version!=Version || b.commands!=16 || b.reserved ||
       b.total_bytes<sizeof(b) || b.total_bytes>capacity || checksum(memory,sizeof(b))!=b.checksum) return error("bundle",2);
    uint64_t end=sizeof(b);
    for(unsigned i=0;i<16;++i) {
        if(b.offsets[i]%8 || b.offsets[i]!=end || b.sizes[i]<sizeof(Command) ||
           b.sizes[i]>MaxCommandBytes || uint64_t(b.offsets[i])+b.sizes[i]>b.total_bytes) return error("bundle",2);
        end=b.offsets[i]+b.sizes[i];
    }
    if(end!=b.total_bytes) return error("bundle",2);
    const unsigned worker=core-4;
    std::vector<uint8_t> command(bytes+b.offsets[worker],bytes+b.offsets[worker]+b.sizes[worker]);
    Command c{};std::memcpy(&c,command.data(),sizeof(c));
    if(validate(c,command.size())!=Status::Success || c.physical_worker!=worker ||
       c.start_cycle || checksum(command.data(),command.size())!=c.checksum) return error("command",3);
    // Runtime-only relocation: existing 2 MiB GM per core, no new tensor buffers.
    asm volatile("rdcycle %0":"=r"(c.start_cycle));
    std::memcpy(command.data(),&c,sizeof(c));
    c.checksum=checksum(command.data(),command.size());
    std::memcpy(command.data(),&c,sizeof(c));
    const uint64_t gm=uint64_t(core)*0x200000ull+0x2000;
    for(size_t i=0;i<command.size();i+=8) {
        uint64_t value;std::memcpy(&value,command.data()+i,8);
        asm volatile(".insn r 0x0b, 7, 0x0f, x0, %0, %1"::"r"(value),"r"(gm+i):"memory");
    }
    const bool probe=std::getenv("GOLEM_NATIVE_ABI_SELFTEST")!=nullptr;
    if(probe) {
        if(wait(999)!=Status::BadToken || submit(gm+8,command.size())!=-int64_t(Status::InvalidDescriptor) ||
           submit(gm,MaxCommandBytes+8)!=-int64_t(Status::InvalidDescriptor)) return error("pre-submit-selftest",6);
        uint64_t original;std::memcpy(&original,command.data(),8);
        const uint64_t unsupported=(original&0xffffffffull)|(uint64_t(99)<<32);
        asm volatile(".insn r 0x0b, 7, 0x0f, x0, %0, %1"::"r"(unsupported),"r"(gm):"memory");
        if(submit(gm,command.size())!=-int64_t(Status::Unsupported)) return error("pre-submit-selftest",6);
        asm volatile(".insn r 0x0b, 7, 0x0f, x0, %0, %1"::"r"(original),"r"(gm):"memory");
        // Corrupt the actual GM command, prove checksum rejection, then restore it.
        uint64_t word;std::memcpy(&word,command.data()+16,8);
        const uint64_t corrupt=word^1;
        asm volatile(".insn r 0x0b, 7, 0x0f, x0, %0, %1"::"r"(corrupt),"r"(gm+16):"memory");
        if(submit(gm,command.size())!=-int64_t(Status::InvalidDescriptor)) return error("pre-submit-selftest",6);
        asm volatile(".insn r 0x0b, 7, 0x0f, x0, %0, %1"::"r"(word),"r"(gm+16):"memory");
    }
    const int64_t token=submit(gm,command.size());
    if(token<=0) {std::printf("[NATIVE_GUEST_ERROR] core=%u submit=%lld\n",core,(long long)token);return 4;}
    if(probe && (submit(gm,command.size())!=-int64_t(Status::QueueFull) || wait(token+1)!=Status::BadToken)) return error("busy-selftest",7);
    const auto status=wait(token);
    if(probe) {
        if(wait(token)!=Status::BadToken) return error("consumed-selftest",8);
        std::printf("[NATIVE_ABI_SELFTEST] core=%u status=PASS\n",core);
    }
    std::printf("[NATIVE_GUEST_DONE] core=%u token=%llu status=%u\n",core,(unsigned long long)token,unsigned(status));
    return status==Status::Success?0:5;
}
}
