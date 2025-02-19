#pragma once
using namespace metal;
#include <metal_stdlib>
#include "../misc/types.h"


kernel void bpr_stage_1(
    constant Jacobian* buckets [[ buffer(0) ]],
    device Jacobian* m_shared [[ buffer(1) ]],
    device Jacobian* s_shared [[ buffer(2) ]],
    constant uint32_t& bucket_size [[ buffer(3) ]],
    constant uint32_t& total_threads [[ buffer(4) ]],
    // TODO: remove me: debugging
    device uint32_t* debug_idx [[ buffer(5) ]],
    device Jacobian* debug_buckets [[ buffer(6) ]],
    device uint32_t& debug_r [[ buffer(7) ]],
    uint gid [[ thread_position_in_grid ]]
);

kernel void bpr_stage_2(
    device Jacobian* result [[ buffer(0) ]],
    device Jacobian* s_shared [[ buffer(1) ]],
    device Jacobian* m_shared [[ buffer(2) ]],
    constant uint32_t& bucket_size [[ buffer(3) ]],
    constant uint32_t& total_threads [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid]]
);


