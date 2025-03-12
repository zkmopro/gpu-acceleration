using namespace metal;
#include <metal_stdlib>
#include "../curve/jacobian.metal"
#include "../misc/get_constant.metal"

kernel void parallel_bpr(
    constant FieldElement* buckets_x         [[ buffer(0) ]],
    constant FieldElement* buckets_y         [[ buffer(1) ]],
    constant FieldElement* buckets_z         [[ buffer(2) ]],
    device FieldElement* m_x                 [[ buffer(3) ]],
    device FieldElement* m_y                 [[ buffer(4) ]],
    device FieldElement* m_z                 [[ buffer(5) ]],
    device FieldElement* s_x                 [[ buffer(6) ]],
    device FieldElement* s_y                 [[ buffer(7) ]],
    device FieldElement* s_z                 [[ buffer(8) ]],
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
            { m_x[gid].value },
            { m_y[gid].value },
            { m_z[gid].value }
        };

        Jacobian s_shared = {
            { s_x[gid].value },
            { s_y[gid].value },
            { s_z[gid].value }
        };

        Jacobian bucket_val;

        if (l != 1) {
            bucket_val.x.value = buckets_x[(gid + 1) * r - l].value;
            bucket_val.y.value = buckets_y[(gid + 1) * r - l].value;
            bucket_val.z.value = buckets_z[(gid + 1) * r - l].value;
            
            m_shared = m_shared + bucket_val;
            s_shared = s_shared + m_shared;
        } else {
            bucket_val.x.value = buckets_x[(gid + 1) * r - 1].value;
            bucket_val.y.value = buckets_y[(gid + 1) * r - 1].value;
            bucket_val.z.value = buckets_z[(gid + 1) * r - 1].value;
            
            m_shared = bucket_val;
            s_shared = m_shared;
        }
        
        m_x[gid].value = m_shared.x.value;
        m_y[gid].value = m_shared.y.value;
        m_z[gid].value = m_shared.z.value;
        
        s_x[gid].value = s_shared.x.value;
        s_y[gid].value = s_shared.y.value;
        s_z[gid].value = s_shared.z.value;
    }

    Jacobian final_s = {
        { s_x[gid].value },
        { s_y[gid].value },
        { s_z[gid].value }
    };

    Jacobian m = {
        { m_x[gid].value },
        { m_y[gid].value },
        { m_z[gid].value }
    };

    uint32_t scalar = gid * r;
    final_s = final_s + jacobian_scalar_mul(m, scalar);
    
    s_x[gid].value = final_s.x.value;
    s_y[gid].value = final_s.y.value;
    s_z[gid].value = final_s.z.value;
}
