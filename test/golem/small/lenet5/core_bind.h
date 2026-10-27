#pragma once

#include <sched.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>

#include "gm_config.h"

inline void bind_process_to_core(int core_id) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);

    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        perror("sched_setaffinity");
    } else {
        DEBUG_PRINT("[Core%d] sched_setaffinity success\n", core_id);
    }
}

inline int resolve_core_id(int requested_core) {
    int actual_core = sched_getcpu();
    if (actual_core < 0) {
        perror("sched_getcpu");
        fprintf(stderr, "ERROR: 无法获取实际核心 ID，sched_getcpu 失败 (errno=%d)\n", errno);
        _exit(1);
    }

    if (requested_core >= 0 && requested_core != actual_core) {
        printf("[WARN] 请求绑定核心 %d，但实际运行在核心 %d。将使用实际核心 ID。\n",
               requested_core, actual_core);
    }

    return actual_core;
}

inline int bind_and_resolve_core_from_argv_or_exit(int argc, char* argv[], int max_cores) {
    int requested_core = -1;
    if (argc >= 2) {
        requested_core = std::atoi(argv[1]);
        if (requested_core < 0 || requested_core >= max_cores) {
            std::fprintf(stderr,
                         "ERROR: core_id=%d 无效，应在 [0, %d) 范围内。\n",
                         requested_core,
                         max_cores);
            _exit(1);
        }
        bind_process_to_core(requested_core);
    }

    int core_id = resolve_core_id(requested_core);
    if (core_id < 0 || core_id >= max_cores) {
        std::fprintf(stderr,
                     "ERROR: 实际核心 ID=%d 越界 (TOTAL_CORES=%d)。\n",
                     core_id,
                     max_cores);
        _exit(1);
    }
    return core_id;
}
