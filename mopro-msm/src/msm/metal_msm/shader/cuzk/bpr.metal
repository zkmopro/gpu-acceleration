#include "bpr.h"
// constant os_log logger(/*subsystem=*/"com.metal.xyz", /*category=*/"abc");

// the first stage of BPR. after this, 
// s_shared[tid] = B_{(tid)r+1} + 2*B_{(tid)r+2} + ... + r*B_{(tid)r+r}
// m_shared[tid*r] = B_{(tid)r+1} + B_{(tid)r+2} + ... + B_{(tid)r+r}
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
) {     
    // first version: 
    if (gid >= total_threads) {
        return;
    }  

    uint32_t tid = gid;
    // Calculate r = (2^c - 1) / total_threads
    uint32_t r = uint32_t(ceil(float(bucket_size) / float(total_threads)));
    
    // TODO: remove me: debugging outputs
    debug_idx[tid] = tid;
    debug_buckets[tid] = buckets[tid];
    debug_r = r;

//    logger.log_error("Hello Not There!");
//    metal::os_log_default.log_debug("Hello There!");

    uint32_t base = tid * r;

    if (base >= bucket_size) {
        s_shared[tid] = jacobian_zero();
        return;
    }

    m_shared[base] = buckets[base];
    s_shared[tid] = buckets[base];

    for (uint32_t l = 1; l < r; l++) {
        uint32_t idx = base + l;
        if (idx >= bucket_size) {
            break;
        }

        m_shared[idx] = m_shared[idx - 1] + buckets[idx];
        s_shared[tid] = s_shared[tid] + m_shared[idx];
    }
}

// the second stage of BPR.
// s_shared[tid] = s_shared[tid] + ((tid - 1) * r) * m_shared[tid*r]
kernel void bpr_stage_2(
    device Jacobian* result [[ buffer(0) ]],
    device Jacobian* s_shared [[ buffer(1) ]],
    device Jacobian* m_shared [[ buffer(2) ]],
    constant uint32_t& bucket_size [[ buffer(3) ]],
    constant uint32_t& total_threads [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid]]
) {
    uint32_t tid = gid + 1;

    if (tid >= bucket_size) {
        return;
    }

    // Calculate r = (2^c - 1) / total_threads
    uint32_t r = bucket_size / total_threads;

    Jacobian m_last = m_shared[tid * r];
    uint32_t scalar = (tid - 1) * r;
    s_shared[tid] = s_shared[tid] + jacobian_scalar_mul(m_last, scalar);

    // Store final result into result
    result[tid] = s_shared[tid];
}
