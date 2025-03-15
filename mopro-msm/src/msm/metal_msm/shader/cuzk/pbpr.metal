using namespace metal;
#include <metal_stdlib>
#include "../curve/jacobian.metal"
#include "../misc/get_constant.metal"

kernel void parallel_bpr(
    constant BigInt* buckets_x         [[ buffer(0) ]],
    constant BigInt* buckets_y         [[ buffer(1) ]],
    constant BigInt* buckets_z         [[ buffer(2) ]],
    device BigInt* m_x                 [[ buffer(3) ]],
    device BigInt* m_y                 [[ buffer(4) ]],
    device BigInt* m_z                 [[ buffer(5) ]],
    device BigInt* s_x                 [[ buffer(6) ]],
    device BigInt* s_y                 [[ buffer(7) ]],
    device BigInt* s_z                 [[ buffer(8) ]],
    constant uint32_t& grid_width      [[ buffer(9) ]],
    constant uint32_t& total_threads   [[ buffer(10) ]],
    constant uint32_t& r               [[ buffer(11) ]],
    uint2 tid                          [[ thread_position_in_grid ]]
) {
    // Convert the 2D thread coordinate into a flat index.
    uint gid = tid.y * grid_width + tid.x;
    if (gid >= total_threads) {
        return;
    }
   
    // Accumulating buckets into s_shared and m_shared using 0-based indexing.
    for (uint32_t l = 1; l <= r; l++) {
        
        Jacobian m_shared = {
            m_x[gid],
            m_y[gid],
            m_z[gid]
        };

        Jacobian s_shared = {
            s_x[gid],
            s_y[gid],
            s_z[gid]
        };

        Jacobian bucket_val;

        if (l != 1) {
            bucket_val.x = buckets_x[(gid + 1) * r - l];
            bucket_val.y = buckets_y[(gid + 1) * r - l];
            bucket_val.z = buckets_z[(gid + 1) * r - l];
            
            m_shared = m_shared + bucket_val;
            s_shared = s_shared + m_shared;
        } else {
            bucket_val.x = buckets_x[(gid + 1) * r - 1];
            bucket_val.y = buckets_y[(gid + 1) * r - 1];
            bucket_val.z = buckets_z[(gid + 1) * r - 1];
            
            m_shared = bucket_val;
            s_shared = m_shared;
        }
        
        m_x[gid] = m_shared.x;
        m_y[gid] = m_shared.y;
        m_z[gid] = m_shared.z;
        
        s_x[gid] = s_shared.x;
        s_y[gid] = s_shared.y;
        s_z[gid] = s_shared.z;
    }

    Jacobian final_s = {
        s_x[gid],
        s_y[gid],
        s_z[gid]
    };

    Jacobian m = {
        m_x[gid],
        m_y[gid],
        m_z[gid]
    };

    uint32_t scalar = gid * r;
    final_s = final_s + jacobian_scalar_mul(m, scalar);
    
    s_x[gid] = final_s.x;
    s_y[gid] = final_s.y;
    s_z[gid] = final_s.z;
}
