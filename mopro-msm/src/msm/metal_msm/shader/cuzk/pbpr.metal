#include "../curve/xyzz.metal"
#include "../misc/get_constant.metal"
#include <metal_stdlib>
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
static XYZZ double_and_add(XYZZ point, uint scalar)
{
    XYZZ result = get_xyzz_zero();

    uint s = scalar;
    XYZZ temp = point;

    while (s != 0u) {
        if ((s & 1u) == 1u) {
            result = xyzz_add_safe(result, temp);
        }
        temp = xyzz_dbl(temp);
        s = s >> 1u;
    }
    return result;
}

kernel void bpr_stage_1(
    device BigInt* bucket_sum_x [[buffer(0), access(read_write)]],
    device BigInt* bucket_sum_y [[buffer(1), access(read_write)]],
    device BigInt* bucket_sum_zz [[buffer(2), access(read_write)]],
    device BigInt* bucket_sum_zzz [[buffer(3), access(read_write)]],
    device BigInt* g_points_x [[buffer(4), access(write)]],
    device BigInt* g_points_y [[buffer(5), access(write)]],
    device BigInt* g_points_zz [[buffer(6), access(write)]],
    device BigInt* g_points_zzz [[buffer(7), access(write)]],
    constant uint3& params [[buffer(8), access(read)]],
    uint tid [[thread_position_in_grid]],
    uint workgroup_size [[dispatch_threads_per_threadgroup]])
{
    const uint thread_id = tid;
    const uint num_threads_per_subtask = workgroup_size;

    const uint subtask_idx = params[0];
    const uint num_columns = params[1];
    const uint num_subtasks_per_bpr = params[2]; // Number of subtasks per shader invocation

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
    if (idx >= total_buckets) {
        return;
    }

    XYZZ m = {
        .x = bucket_sum_x[idx],
        .y = bucket_sum_y[idx],
        .zz = bucket_sum_zz[idx],
        .zzz = bucket_sum_zzz[idx]
    };
    XYZZ g = m;

    for (uint i = 0; i < buckets_per_thread - 1u; i++) {
        uint idx = (num_threads_per_subtask - (thread_id % num_threads_per_subtask)) * buckets_per_thread - 1u - i;
        uint bi = offset + idx;
        XYZZ b = {
            .x = bucket_sum_x[bi],
            .y = bucket_sum_y[bi],
            .zz = bucket_sum_zz[bi],
            .zzz = bucket_sum_zzz[bi]
        };
        m = xyzz_add_safe(m, b);
        g = xyzz_add_safe(g, m);
    }

    bucket_sum_x[idx] = m.x;
    bucket_sum_y[idx] = m.y;
    bucket_sum_zz[idx] = m.zz;
    bucket_sum_zzz[idx] = m.zzz;

    uint g_rw_idx = (subtask_idx / num_subtasks_per_bpr) * (num_threads_per_subtask * num_subtasks_per_bpr) + thread_id;
    g_points_x[g_rw_idx] = g.x;
    g_points_y[g_rw_idx] = g.y;
    g_points_zz[g_rw_idx] = g.zz;
    g_points_zzz[g_rw_idx] = g.zzz;
}

kernel void bpr_stage_2(
    device BigInt* bucket_sum_x [[buffer(0), access(read)]],
    device BigInt* bucket_sum_y [[buffer(1), access(read)]],
    device BigInt* bucket_sum_zz [[buffer(2), access(read)]],
    device BigInt* bucket_sum_zzz [[buffer(3), access(read)]],
    device BigInt* g_points_x [[buffer(4), access(read_write)]],
    device BigInt* g_points_y [[buffer(5), access(read_write)]],
    device BigInt* g_points_zz [[buffer(6), access(read_write)]],
    device BigInt* g_points_zzz [[buffer(7), access(read_write)]],
    constant uint3& params [[buffer(8), access(read)]],
    uint tid [[thread_position_in_grid]],
    uint workgroup_size [[dispatch_threads_per_threadgroup]])
{
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

    XYZZ m = {
        .x = bucket_sum_x[idx],
        .y = bucket_sum_y[idx],
        .zz = bucket_sum_zz[idx],
        .zzz = bucket_sum_zzz[idx]
    };

    const uint g_rw_idx = (subtask_idx / num_subtasks_per_bpr) * (num_threads_per_subtask * num_subtasks_per_bpr) + thread_id;
    XYZZ g = {
        .x = g_points_x[g_rw_idx],
        .y = g_points_y[g_rw_idx],
        .zz = g_points_zz[g_rw_idx],
        .zzz = g_points_zzz[g_rw_idx]
    };

    // Perform scalar mul on m and add the result to g
    const uint s = buckets_per_thread * (num_threads_per_subtask - (thread_id % num_threads_per_subtask) - 1u);
    g = xyzz_add_safe(g, double_and_add(m, s));

    g_points_x[g_rw_idx] = g.x;
    g_points_y[g_rw_idx] = g.y;
    g_points_zz[g_rw_idx] = g.zz;
    g_points_zzz[g_rw_idx] = g.zzz;
}

// Stage 3: collapse each subtask's g_points into a single point so the host
// only Horner-combines num_subtasks points. One threadgroup per subtask;
// tree reduction in threadgroup memory (128 * sizeof(XYZZ) = 16 KB).
#define BPR_REDUCE_THREADS 128u

kernel void bpr_reduce(
    device const BigInt* g_points_x [[buffer(0), access(read)]],
    device const BigInt* g_points_y [[buffer(1), access(read)]],
    device const BigInt* g_points_zz [[buffer(2), access(read)]],
    device const BigInt* g_points_zzz [[buffer(3), access(read)]],
    device BigInt* subtask_x [[buffer(4), access(write)]],
    device BigInt* subtask_y [[buffer(5), access(write)]],
    device BigInt* subtask_zz [[buffer(6), access(write)]],
    device BigInt* subtask_zzz [[buffer(7), access(write)]],
    constant uint2& params [[buffer(8), access(read)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const uint points_per_subtask = params[0];

    threadgroup XYZZ shared_pts[BPR_REDUCE_THREADS];

    // Grid-stride accumulate this thread's share of the subtask's points
    XYZZ acc = get_xyzz_zero();
    for (uint k = tid; k < points_per_subtask; k += BPR_REDUCE_THREADS) {
        const uint gi = tgid * points_per_subtask + k;
        XYZZ p = {
            .x = g_points_x[gi],
            .y = g_points_y[gi],
            .zz = g_points_zz[gi],
            .zzz = g_points_zzz[gi]
        };
        acc = xyzz_add_safe(acc, p);
    }
    shared_pts[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = BPR_REDUCE_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_pts[tid] = xyzz_add_safe(shared_pts[tid], shared_pts[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        subtask_x[tgid] = shared_pts[0].x;
        subtask_y[tgid] = shared_pts[0].y;
        subtask_zz[tgid] = shared_pts[0].zz;
        subtask_zzz[tgid] = shared_pts[0].zzz;
    }
}
