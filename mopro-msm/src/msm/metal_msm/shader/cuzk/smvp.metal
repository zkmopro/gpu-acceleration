#include <metal_stdlib>
#include "barrett_reduction.metal"
#include "../curve/jacobian.metal"
using namespace metal;

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
    #include <metal_logging>
    constant os_log smvp_logger_kernel(/*subsystem=*/"smvp", /*category=*/"metal");
    #define LOG_DEBUG(...) smvp_logger_kernel.log_debug(__VA_ARGS__)
#else
    #define LOG_DEBUG(...) ((void)0)
#endif

kernel void smvp(
    device const uint*          row_ptr             [[ buffer(0) ]],
    device const uint*          val_idx             [[ buffer(1) ]],
    device const BigInt*        new_point_x         [[ buffer(2) ]],
    device const BigInt*        new_point_y         [[ buffer(3) ]],
    device BigInt*              bucket_x            [[ buffer(4) ]],
    device BigInt*              bucket_y            [[ buffer(5) ]],
    device BigInt*              bucket_z            [[ buffer(6) ]],
    constant uint4&             params              [[ buffer(7) ]],
    uint3                       tgid                [[ threadgroup_position_in_grid ]],
    uint3                       tid                 [[ thread_position_in_threadgroup ]],
    uint3                       workgroup_size      [[ dispatch_threads_per_threadgroup ]],
    uint3                       threadgroup_size    [[ threadgroups_per_grid ]]
) {
    const uint input_size        = params[0];
    const uint num_columns       = params[1];
    const uint num_subtasks      = params[2];
    const uint subtask_offset    = params[3];

    const uint tgidx = tgid.x;
    const uint tgidy = tgid.y;
    const uint tgidz = tgid.z;

    const uint group_id = (tgidx * threadgroup_size.y + tgidy) * threadgroup_size.z + tgidz;
    const uint id = group_id * workgroup_size.x + tid.x;

    const uint half_columns = num_columns / 2;

    const uint subtask_idx = id / half_columns;

    // Add bounds checking to prevent out-of-bounds access
    if (subtask_idx + subtask_offset >= num_subtasks) {
        LOG_DEBUG("tgidx %u, tgidy %u, tgidz %u, tid.x %u, workgroup_size.x %u, group_id %u, id %u, subtask_idx %u, subtask_offset %u", tgidx, tgidy, tgidz, tid.x, workgroup_size.x, group_id, id, subtask_idx, subtask_offset);
        return;
    }

    Jacobian inf = get_bn254_zero_mont();

    // an offset for each subtask's row_ptr
    const uint rp_offset = (subtask_idx + subtask_offset) * (num_columns + 1);

    // Each thread handles two buckets (one positive, one negative)
    for (uint j = 0; j < 2; j++) {
        uint row_idx = (id % half_columns) + half_columns;
        if (j == 1) {
            row_idx = half_columns - (id % half_columns);
        }
        if (j == 0 && (id % half_columns) == 0) {
            row_idx = 0;
        }

        // Add bounds checking for row_ptr access
        if (rp_offset + row_idx + 1 >= (16 * (num_columns + 1) * 4)) { // Total row_ptr buffer size
            LOG_DEBUG("SMVP: row_ptr bounds exceeded at offset %u, row_idx %u", rp_offset, row_idx);
            return;
        }

        const uint row_begin = row_ptr[rp_offset + row_idx];
        const uint row_end = row_ptr[rp_offset + row_idx + 1];

        // Add safety check to prevent infinite loops
        if (row_end > input_size || row_begin > row_end) {
            LOG_DEBUG("SMVP: Invalid row range [%u, %u) for subtask %u, offset %u", 
                     row_begin, row_end, subtask_idx, subtask_offset);
            continue; // Skip this bucket instead of hanging
        }

        // Add another safety check to prevent very large loops
        if (row_end - row_begin > input_size / 2) {
            LOG_DEBUG("SMVP: Suspiciously large bucket size %u for subtask %u", 
                     row_end - row_begin, subtask_idx + subtask_offset);
            continue; // Skip this bucket
        }

        // Accumulate all the points for that bucket
        Jacobian sum = inf;

        LOG_DEBUG("SMVP: row_begin %u, row_end %u, subtask_idx %u, subtask_offset %u", row_begin, row_end, subtask_idx, subtask_offset);
        if (row_begin > row_end) {
            LOG_DEBUG("SMVP: row_begin %u is greater than row_end %u", row_begin, row_end);
            continue;
        }

        for (uint k = row_begin; k < row_end; k++) {
            // Add bounds checking for val_idx access
            const uint val_idx_offset = (subtask_idx + subtask_offset) * input_size + k;
            if (val_idx_offset >= (16 * input_size)) { // Total val_idx buffer size
                LOG_DEBUG("SMVP: val_idx bounds exceeded at offset %u", val_idx_offset);
                break;
            }
            
            const uint idx = val_idx[val_idx_offset];
            
            // Add bounds checking for point array access
            if (idx >= input_size) {
                LOG_DEBUG("SMVP: Point index %u exceeds input_size %u", idx, input_size);
                continue;
            }
            
            Jacobian b = {
                .x = new_point_x[idx],
                .y = new_point_y[idx],
                .z = get_bn254_one_mont().z
            };
            sum = sum + b;
        }

        // In short Weierstrass, negation = flip sign of Y mod p: jacobian_neg.
        uint bucket_idx = 0;
        // Negative bucket
        if (half_columns > row_idx) {
            bucket_idx = half_columns - row_idx;
            sum = -sum;
        }
        // Positive bucket
        else {
            bucket_idx = row_idx - half_columns;
        }

        // Store the result in bucket arrays only if bucket_idx > 0
        // The final 1D index for the bucket is id + subtask_offset * half_columns
        const uint bi = id + subtask_offset * half_columns;
        
        // Add bounds checking for bucket array access
        const uint bucket_size = half_columns * NUM_LIMBS * num_subtasks;
        if (bi >= bucket_size) {
            LOG_DEBUG("SMVP: Bucket index %u exceeds buffer size %u", bi, bucket_size);
            return;
        }
        
        if (bucket_idx > 0) {
            // If j == 1, add to the existing bucket at index `bi`.
            if (j == 1) {
                // Load the previous partial sum from bucket_{x,y,z}, interpret as Jacobian
                Jacobian bucket_val = {
                    .x = bucket_x[bi],
                    .y = bucket_y[bi],
                    .z = bucket_z[bi]
                };
                sum = bucket_val + sum;
            }
            bucket_x[bi] = sum.x;
            bucket_y[bi] = sum.y;
            bucket_z[bi] = sum.z;
        }
    }
}
