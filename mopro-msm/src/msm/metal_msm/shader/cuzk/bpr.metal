using namespace metal;
#include <metal_stdlib>
#include "../curve/jacobian.metal"
#include "../misc/get_constant.metal"

kernel void bpr(
    constant Jacobian* buckets [[ buffer(0) ]],
    device Jacobian* result [[ buffer(1) ]],
    constant uint32_t& bucket_size [[ buffer(2) ]],
    constant uint32_t& total_threads [[ buffer(3) ]], device Jacobian* m_shared [[ buffer(5) ]],
    device Jacobian* s_shared [[ buffer(6) ]],
    uint gid [[ thread_position_in_grid ]]) {
      
    if (gid >= total_threads) {
      return;
    }  

    // Calculate r = (2^c - 1) / total_threads
    uint32_t r = bucket_size / total_threads;

    // Accumulating buckets to s_shared and m_shared
    for (uint32_t i = 1; i <= r; i++) {
        uint32_t m_idx = (gid - 1) * r + i;  

        if (i != 1) {
            m_shared[m_idx] = m_shared[m_idx - 1] + buckets[gid * r + 1 - i];
            Jacobian current_s = s_shared[gid];
            s_shared[gid] = current_s + m_shared[m_idx];
        } else {
            m_shared[m_idx] = buckets[gid * r];  
            s_shared[gid] = m_shared[m_idx];
        }
    }

    // After the above for loop
    // s_shared[tid] = B_{(tid-1)r+1} + 2*B_{(tid-1)r+2} + ... + r*B_{(tid-1)r+r}
    // m_shared[tid*r] = B_{(t-1)r+1} + B_{(tid-1)r+2} + ... + B_{(tid-1)r+r}

    Jacobian m_last = m_shared[gid * r];  
    uint32_t scalar = (gid - 1) * r;
    Jacobian scaled_m = jacobian_scalar_mul(m_last, scalar);
    s_shared[gid] = s_shared[gid] + scaled_m;

    result[gid] = s_shared[gid];
}
