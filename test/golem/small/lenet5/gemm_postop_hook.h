#pragma once

#include <cstddef>

#include "pipeline_config.h"

struct GolemFp32TilePostOpHook {
    void (*fn)(const GemmTaskDescriptor& desc, float* c_tile, size_t elems, void* user);
    void* user;
};

GolemFp32TilePostOpHook golem_get_fp32_tile_postop_hook();
void golem_set_fp32_tile_postop_hook(GolemFp32TilePostOpHook hook);
