#pragma once

/* CPU competitors */
#include "competitors/cpu/cpu_basic.hpp"
#include "competitors/cpu/cpu_pytorch.hpp"


/* GPU competitors */
#include "competitors/gpu/gpu_basic.hpp"
#include "competitors/gpu/gpu_pytorch.hpp"
#include "competitors/gpu/gpu_tiled.hpp"
#include "competitors/gpu/gpu_adaptive_tiling.hpp"
#include "competitors/gpu/gpu_thread_dispatcher.hpp"
#include "competitors/gpu/gpu_shared.hpp"
