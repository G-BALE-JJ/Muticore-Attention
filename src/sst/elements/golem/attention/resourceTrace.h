#ifndef SST_GOLEM_RESOURCE_TRACE_H
#define SST_GOLEM_RESOURCE_TRACE_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace SST { namespace Golem {

// One completed interval, in SST core ticks, at the resource that actually
// schedules it. IDs are scoped by (core, domain, resource). No model timing,
// request IDs, callback order, or payload storage is changed by tracing.
inline bool resourceTraceEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("GOLEM_CRITICAL_PATH_TRACE");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

inline void traceResourceInterval(const char* domain, const char* resource,
        int64_t core, uint64_t request, uint64_t tag, uint64_t generation,
        uint64_t submit, uint64_t start, uint64_t done, uint64_t visible,
        uint64_t bytes, uint64_t depth, uint64_t address = 0,
        int64_t client = -1, int64_t traffic = -1, bool ok = true) {
    if (!resourceTraceEnabled()) return;
    // Optional output filtering retains phase/control records and permits
    // experiment matrices without tens of GiB of per-lane SFU/HBM logs.
    static const std::string domains = []() {
        const char* value = std::getenv("GOLEM_RESOURCE_TRACE_DOMAINS");
        return value ? std::string(",") + value + "," : std::string();
    }();
    if (!domains.empty() && domains.find(std::string(",") + domain + ",") == std::string::npos) return;
    std::printf("[RESOURCE_INTERVAL] domain=%s resource=%s core=%lld request=%llu tag=%llu generation=%llu submit_tick=%llu start_tick=%llu done_tick=%llu visible_tick=%llu bytes=%llu queue_depth=%llu address=%llu client=%lld traffic=%lld ok=%u\n",
        domain, resource, (long long)core, (unsigned long long)request,
        (unsigned long long)tag, (unsigned long long)generation,
        (unsigned long long)submit, (unsigned long long)start,
        (unsigned long long)done, (unsigned long long)visible,
        (unsigned long long)bytes, (unsigned long long)depth,
        (unsigned long long)address, (long long)client, (long long)traffic,
        ok ? 1u : 0u);
}

} }
#endif
