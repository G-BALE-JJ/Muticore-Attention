#include <runtime.h>
#include <cstdio>
#include <cstdlib>
#include <sched.h>
int main(int argc,char** argv) {
    if(argc!=2) {std::printf("[NATIVE_GUEST_ERROR] argc=%d\n",argc);return 1;}
    char* end=nullptr;long core=std::strtol(argv[1],&end,10);
    if(!end || *end || core<0 || core>=20) return 1;
    // Vanadis records affinity but does not migrate an already running process.
    // Process argument order is not physical-core order: use getcpu, as the
    // production Attention guest does, and pin to the core actually executing.
    core=sched_getcpu();
    if(core<0 || core>=20) {std::printf("[NATIVE_GUEST_ERROR] invalid physical core=%ld\n",core);return 1;}
    cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(core,&mask);
    if(sched_setaffinity(0,sizeof(mask),&mask)!=0) {std::printf("[NATIVE_GUEST_ERROR] core=%ld affinity failed\n",core);return 1;}
    if(core<4) return 0;
    return GolemNative::runForCore(core,reinterpret_cast<const void*>(GolemNative::VirtualAddress),GolemNative::RegionBytes);
}
