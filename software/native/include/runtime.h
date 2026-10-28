#ifndef GOLEM_NATIVE_GUEST_RUNTIME_H
#define GOLEM_NATIVE_GUEST_RUNTIME_H
#include <sst/elements/golem/isa/nativeDescriptor.h>
namespace GolemNative {
using Token=uint64_t;
// submit uses release ordering; wait provides acquire ordering after final C ACK.
int64_t submit(uint64_t gm_address,uint32_t bytes);
Status wait(Token token);
int runForCore(uint32_t core,const void* bundle,size_t capacity);
}
#endif
