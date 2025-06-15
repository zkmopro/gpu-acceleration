#include <metal_stdlib>
#include "../curve/jacobian.metal"
#include "../misc/get_constant.metal"
using namespace metal;

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
    #include <metal_logging>
    constant os_log pbpr_logger_kernel(/*subsystem=*/"pbpr", /*category=*/"metal");
    #define LOG_DEBUG(...) pbpr_logger_kernel.log_debug(__VA_ARGS__)
#else
    #define LOG_DEBUG(...) ((void)0)
#endif

// This double-and-add code is adapted from the ZPrize test harness:
// https://github.com/demox-labs/webgpu-msm/blob/main/src/reference/webgpu/wgsl/Curve.ts#L78.
static Jacobian double_and_add(Jacobian point, uint scalar) {
    Jacobian result = get_bn254_zero_mont(); // Point at infinity

    uint s = scalar;
    Jacobian temp = point;

    while (s != 0u) {
        if ((s & 1u) == 1u) {
            result = result + temp;
        }
        temp = jacobian_dbl_2009_l(temp);
        s = s >> 1u;
    }
    return result;
}

kernel void bpr_stage_1(
    device BigInt* bucket_sum_x         [[ buffer(0), access(read_write) ]],
    device BigInt* bucket_sum_y         [[ buffer(1), access(read_write) ]],
    device BigInt* bucket_sum_z         [[ buffer(2), access(read_write) ]],
    device BigInt* g_points_x           [[ buffer(3), access(write) ]],
    device BigInt* g_points_y           [[ buffer(4), access(write) ]],
    device BigInt* g_points_z           [[ buffer(5), access(write) ]],
    constant uint3& params              [[ buffer(6), access(read) ]],
    uint tid                            [[ thread_position_in_grid ]],
    uint workgroup_size                 [[ dispatch_threads_per_threadgroup ]]
) {
    const uint thread_id = tid;
    const uint num_threads_per_subtask = workgroup_size;

    const uint subtask_idx = params[0];
    const uint num_columns = params[1];
    const uint num_subtasks_per_bpr = params[2];    // Number of subtasks per shader invocation

    const uint num_buckets_per_subtask = num_columns / 2u;
    const uint total_buckets = num_buckets_per_subtask * num_subtasks_per_bpr;

    // Number of buckets to reduce per thread.
    const uint buckets_per_thread = num_buckets_per_subtask / num_threads_per_subtask;
    const uint multiplier = subtask_idx + (thread_id / num_threads_per_subtask);
    const uint offset = num_buckets_per_subtask * multiplier;

    uint idx = offset;

    if (thread_id % num_threads_per_subtask != 0u) {
        idx = (num_threads_per_subtask - (thread_id % num_threads_per_subtask)) * buckets_per_thread + offset;
    }
    // guard bucket bounds
    if (idx >= total_buckets) { return; }

    Jacobian m = {
        .x = bucket_sum_x[idx],
        .y = bucket_sum_y[idx],
        .z = bucket_sum_z[idx]
    };
    Jacobian g = m;

    for (uint i = 0; i < buckets_per_thread - 1u; i++) {
        uint idx = (num_threads_per_subtask - (thread_id % num_threads_per_subtask)) * buckets_per_thread - 1u - i;
        uint bi = offset + idx;
        Jacobian b = {
            .x = bucket_sum_x[bi],
            .y = bucket_sum_y[bi],
            .z = bucket_sum_z[bi]
        };
        m = m + b;
        g = g + m;
    }

    bucket_sum_x[idx] = m.x;
    bucket_sum_y[idx] = m.y;
    bucket_sum_z[idx] = m.z;

    uint g_rw_idx = (subtask_idx / num_subtasks_per_bpr) * (num_threads_per_subtask * num_subtasks_per_bpr) + thread_id;
    g_points_x[g_rw_idx] = g.x;
    g_points_y[g_rw_idx] = g.y;
    g_points_z[g_rw_idx] = g.z;
}


kernel void bpr_stage_2(
    device BigInt* bucket_sum_x         [[ buffer(0), access(read) ]],
    device BigInt* bucket_sum_y         [[ buffer(1), access(read) ]],
    device BigInt* bucket_sum_z         [[ buffer(2), access(read) ]],
    device BigInt* g_points_x           [[ buffer(3), access(read_write) ]],
    device BigInt* g_points_y           [[ buffer(4), access(read_write) ]],
    device BigInt* g_points_z           [[ buffer(5), access(read_write) ]],
    constant uint3& params              [[ buffer(6), access(read) ]],
    uint tid                            [[ thread_position_in_grid ]],
    uint workgroup_size                 [[ dispatch_threads_per_threadgroup ]]
) {
    const uint thread_id = tid;
    const uint num_threads_per_subtask = workgroup_size;

    const uint subtask_idx = params[0];
    const uint num_columns = params[1];
    const uint num_subtasks_per_bpr = params[2];

    const uint num_buckets_per_subtask = num_columns / 2u;

    const uint buckets_per_thread = num_buckets_per_subtask / num_threads_per_subtask;

    const uint multiplier = subtask_idx + (thread_id / num_threads_per_subtask);
    const uint offset = num_buckets_per_subtask * multiplier;

    uint idx = offset;
    if (thread_id % num_threads_per_subtask != 0u) {
        idx = (num_threads_per_subtask - (thread_id % num_threads_per_subtask)) * buckets_per_thread + offset;
    }

    Jacobian m = {
        .x = bucket_sum_x[idx],
        .y = bucket_sum_y[idx],
        .z = bucket_sum_z[idx]
    };
    
    const uint g_rw_idx = (subtask_idx / num_subtasks_per_bpr) * (num_threads_per_subtask * num_subtasks_per_bpr) + thread_id;
    Jacobian g = {
        .x = g_points_x[g_rw_idx],
        .y = g_points_y[g_rw_idx],
        .z = g_points_z[g_rw_idx]
    };

    // Perform scalar mul on m and add the result to g
    const uint s = buckets_per_thread * (num_threads_per_subtask - (thread_id % num_threads_per_subtask) - 1u);
    g = g + double_and_add(m, s);

    g_points_x[g_rw_idx] = g.x;
    g_points_y[g_rw_idx] = g.y;
    g_points_z[g_rw_idx] = g.z;
}
