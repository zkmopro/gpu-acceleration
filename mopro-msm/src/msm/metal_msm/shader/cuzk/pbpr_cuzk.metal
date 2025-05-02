using namespace metal;
#include <metal_stdlib>
#include "../curve/jacobian.metal"
#include "../misc/get_constant.metal"

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
    #include <metal_logging>
    // Create our real logger.
    constant os_log logger_kernel(/*subsystem=*/"pbpr", /*category=*/"metal");
    // Define the log macro to forward to logger_kernel.log_debug.
    #define LOG_DEBUG(...) logger_kernel.log_debug(__VA_ARGS__)
#else
    // For older Metal versions, define a dummy macro that does nothing.
    #define LOG_DEBUG(...) ((void)0)
#endif

// Basic double-and-add scalar multiplication for Jacobian points
// Note: A more optimized version (like Montgomery ladder or using jacobian_scalar_mul if available) might be preferable.
static Jacobian double_and_add(Jacobian point, uint scalar) {
    Jacobian result = get_bn254_zero_mont(); // Point at infinity

    Jacobian temp = point;
    uint s = scalar;

    // Decompose scalar bit by bit
    while (s != 0u) {
        if ((s & 1u) == 1u) {
            result = jacobian_add_2007_bl(result, temp);
        }
        temp = jacobian_dbl_2009_l(temp);
        s = s >> 1u;
    }
    return result;
}


// Corresponds to stage_1 in the WGSL code
kernel void bpr_stage_1(
    // Bucket sum buffers (read/write, repurposed for m points)
    device BigInt* bucket_sum_x         [[ buffer(0) ]],
    device BigInt* bucket_sum_y         [[ buffer(1) ]],
    device BigInt* bucket_sum_z         [[ buffer(2) ]], // Using Z for Jacobian
    // Output g points buffers
    device BigInt* g_points_x           [[ buffer(3) ]],
    device BigInt* g_points_y           [[ buffer(4) ]],
    device BigInt* g_points_z           [[ buffer(5) ]],
    // Uniform parameters: [subtask_idx, num_columns, num_subtasks_per_bpr, total_subtasks]
    constant packed_uint4& params       [[ buffer(6) ]],
    // Workgroup size passed as a uniform
    constant uint& workgroup_size       [[ buffer(7) ]],
    // Thread ID
    uint tid                            [[ thread_position_in_grid ]]) // Assuming 1D grid
{
    uint thread_id = tid;
    uint num_threads_per_subtask = workgroup_size; // WGSL workgroup_size

    uint subtask_idx = params[0];
    uint num_columns = params[1];
    uint num_subtasks_per_bpr = params[2];
    uint total_subtasks = params[3];

    // Determine which subtask this thread processes
    uint local_subtask_idx = subtask_idx + (thread_id / num_threads_per_subtask);

    // Skip if we’re outside the valid range
    if (local_subtask_idx >= total_subtasks) {
        return;
    }

    // Calculations based on WGSL logic
    uint num_buckets_per_subtask = num_columns / 2u;
    // Ensure num_buckets_per_subtask is divisible by num_threads_per_subtask
    // metal::assert(num_buckets_per_subtask % num_threads_per_subtask == 0);
    uint buckets_per_thread = num_buckets_per_subtask / num_threads_per_subtask;

    // Determine the range of buckets this thread is responsible for within its subtask group
    uint local_thread_idx = thread_id % num_threads_per_subtask;
    uint subtask_group_offset = (subtask_idx + (thread_id / num_threads_per_subtask)) * num_buckets_per_subtask;

    // WGSL processes buckets in reverse order within the thread's assigned chunk.
    // Calculate the start (lowest index) and end (highest index) of the bucket range for this thread.
    uint end_index_local   = (num_threads_per_subtask - local_thread_idx) * buckets_per_thread;
    uint start_index_local = end_index_local - buckets_per_thread;

    uint global_start_index = subtask_group_offset + start_index_local; // Index where m will be stored
    uint first_load_index = subtask_group_offset + end_index_local - 1;  // Highest index to load first

    // Load the last bucket sum in the thread's range
    Jacobian m = { bucket_sum_x[first_load_index], bucket_sum_y[first_load_index], bucket_sum_z[first_load_index] };
    Jacobian g = m; // Initialize g with the first loaded bucket

    // Iterate backwards through the remaining buckets assigned to this thread
    for (uint i = 1; i < buckets_per_thread; ++i) {
        uint current_load_idx = first_load_index - i;
        Jacobian b = { bucket_sum_x[current_load_idx], bucket_sum_y[current_load_idx], bucket_sum_z[current_load_idx] };
        m = jacobian_add_2007_bl(m, b); // Accumulate m
        g = jacobian_add_2007_bl(g, m); // Accumulate g (sum of m's)
    }

    // Store the final accumulated m value back into the bucket sum buffer
    // at the start index of this thread's range.
    bucket_sum_x[global_start_index] = m.x;
    bucket_sum_y[global_start_index] = m.y;
    bucket_sum_z[global_start_index] = m.z;

    // Calculate the global index for writing the final g point
    uint g_write_idx = (subtask_idx / num_subtasks_per_bpr) * (num_threads_per_subtask * num_subtasks_per_bpr) + thread_id;
    g_points_x[g_write_idx] = g.x;
    g_points_y[g_write_idx] = g.y;
    g_points_z[g_write_idx] = g.z;
}


// Corresponds to stage_2 in the WGSL code
kernel void bpr_stage_2(
    // Input m points buffers (previously bucket_sum, read-only)
    device BigInt* m_points_x           [[ buffer(0) ]], // Renamed for clarity
    device BigInt* m_points_y           [[ buffer(1) ]],
    device BigInt* m_points_z           [[ buffer(2) ]],
    // Output g points buffers (read/write)
    device BigInt* g_points_x           [[ buffer(3) ]],
    device BigInt* g_points_y           [[ buffer(4) ]],
    device BigInt* g_points_z           [[ buffer(5) ]],
    // Uniform parameters: [subtask_idx, num_columns, num_subtasks_per_bpr, total_subtasks]
    constant packed_uint4& params       [[ buffer(6) ]],
    // Workgroup size passed as a uniform
    constant uint& workgroup_size       [[ buffer(7) ]],
    // Thread ID
    uint tid                            [[ thread_position_in_grid ]]) // Assuming 1D grid
{
    uint thread_id = tid;
    uint num_threads_per_subtask = workgroup_size;

    uint subtask_idx = params[0];
    uint num_columns = params[1];
    uint num_subtasks_per_bpr = params[2];
    uint total_subtasks = params[3];

    // Determine which subtask this thread processes
    uint local_subtask_idx = subtask_idx + (thread_id / num_threads_per_subtask);

    // Skip if we’re outside the valid range
    if (local_subtask_idx >= total_subtasks) {
        return;
    }

    uint num_buckets_per_subtask = num_columns / 2u;
    // metal::assert(num_buckets_per_subtask % num_threads_per_subtask == 0);
    uint buckets_per_thread = num_buckets_per_subtask / num_threads_per_subtask;

    // Determine where this thread stored its 'm' value in stage 1
    uint local_thread_idx = thread_id % num_threads_per_subtask;
    uint subtask_group_offset = (subtask_idx + (thread_id / num_threads_per_subtask)) * num_buckets_per_subtask;

    // Calculate the start index local to the subtask where m was stored
    uint end_index_local   = (num_threads_per_subtask - local_thread_idx) * buckets_per_thread;
    uint start_index_local = end_index_local - buckets_per_thread;
    uint m_load_idx = subtask_group_offset + start_index_local; // Global index to load m from

    // Load the m point calculated by this thread in stage 1
    Jacobian m = { m_points_x[m_load_idx], m_points_y[m_load_idx], m_points_z[m_load_idx] };

    // Calculate the global index for reading/writing the g point
    uint g_rw_idx = (subtask_idx / num_subtasks_per_bpr) * (num_threads_per_subtask * num_subtasks_per_bpr) + thread_id;

    // Load the g point partially computed in stage 1
    Jacobian g = { g_points_x[g_rw_idx], g_points_y[g_rw_idx], g_points_z[g_rw_idx] };

    // Calculate the scalar 's' based on the WGSL logic
    // s = buckets_per_thread * (num_threads_per_subtask - (thread_id % num_threads_per_subtask) - 1u);
    uint s = buckets_per_thread * (num_threads_per_subtask - local_thread_idx - 1u);
    // LOG_DEBUG("s: %u", s);

    // Perform scalar multiplication [s]m and add the result to g
    // g = g + [s]m
    if (s > 0) { // Optimization: Adding [0]P = O (identity) is a no-op
       Jacobian m_scaled = double_and_add(m, s);
       g = jacobian_add_2007_bl(g, m_scaled);
    }
    // If s == 0, g remains unchanged.

    // Store the final g point result
    g_points_x[g_rw_idx] = g.x;
    g_points_y[g_rw_idx] = g.y;
    g_points_z[g_rw_idx] = g.z;
}
