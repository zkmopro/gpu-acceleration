using namespace metal;
#include <metal_stdlib>
#include "../curve/jacobian.metal"
#include "../misc/get_constant.metal"

kernel void parallel_bpr(
    constant Jacobian* buckets [[ buffer(0) ]],
    device Jacobian* m_shared [[ buffer(1) ]],
    device Jacobian* s_shared [[ buffer(2) ]],
    constant uint32_t& bucket_size [[ buffer(3) ]],
    constant uint32_t& total_threads [[ buffer(4) ]],
    constant uint32_t& r [[ buffer(5) ]],
    uint gid [[ thread_position_in_grid ]]
) {     
    // first version: 
    if (gid >= total_threads) {
        return;
    }

    // Accumulating buckets to s_shared and m_shared using 0-based indexing
    for (uint32_t l = 1; l <= r; l++) {
        if (l != 1) {
            m_shared[gid] = m_shared[gid] + buckets[(gid + 1) * r - l];
            s_shared[gid] = s_shared[gid] + m_shared[gid];
        } else {
            m_shared[gid] = buckets[(gid + 1) * r - 1];
            s_shared[gid] = m_shared[gid];
        }
    }
    uint32_t scalar = gid * r;
    s_shared[gid] = s_shared[gid] + jacobian_scalar_mul(m_shared[gid], scalar);
}