#pragma once

/* CPU competitors */
#include "competitors/cpu/cpu_basic.hpp"
#include "competitors/cpu/cpu_pytorch.hpp"


/* GPU competitors */
#include "competitors/gpu/gpu_basic.hpp"
#include "competitors/gpu/gpu_pytorch.hpp"
#include "competitors/gpu/gpu_tiled.hpp"
#include "competitors/gpu/gpu_thread_dispatcher.hpp"
#include "competitors/gpu/gpu_tensor.hpp"
#include "competitors/gpu/gpu_shared.hpp"
#include "competitors/gpu/gpu_convert.hpp"
#include "competitors/gpu/gpu_preprocessing.hpp"
#include "competitors/gpu/gpu_dynamic.hpp"
#include "competitors/gpu/gpu_cuSPARSE.hpp"
