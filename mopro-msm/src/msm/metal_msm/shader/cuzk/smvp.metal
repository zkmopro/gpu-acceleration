#include "../curve/jacobian.metal"
#include "barrett_reduction.metal"
#include <metal_stdlib>
using namespace metal;

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
#include <metal_logging>
constant os_log smvp_logger_kernel(/*subsystem=*/"smvp", /*category=*/"metal");
#define LOG_DEBUG(...) smvp_logger_kernel.log_debug(__VA_ARGS__)
#else
#define LOG_DEBUG(...) ((void)0)
#endif

kernel void smvp(
    device const uint* row_ptr [[buffer(0), access(read)]],
    device const uint* val_idx [[buffer(1), access(read)]],
    device const BigInt* new_point_x [[buffer(2), access(read)]],
    device const BigInt* new_point_y [[buffer(3), access(read)]],
    device BigInt* bucket_x [[buffer(4), access(read_write)]],
    device BigInt* bucket_y [[buffer(5), access(read_write)]],
    device BigInt* bucket_z [[buffer(6), access(read_write)]],
    constant uint4& params [[buffer(7), access(read)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 workgroup_size [[dispatch_threads_per_threadgroup]],
    uint3 threadgroup_size [[threadgroups_per_grid]])
{
    const uint group_id = (tgid.x * threadgroup_size.y + tgid.y) * threadgroup_size.z + tgid.z;
    const uint id = group_id * workgroup_size.x + tid.x;

    const uint input_size = params[0];
    const uint num_columns = params[1];
    const uint num_subtasks = params[2];
    const uint subtask_offset = params[3];

    const uint half_columns = num_columns / 2;

    const uint subtask_idx = id / half_columns;

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

        const uint row_begin = row_ptr[rp_offset + row_idx];
        const uint row_end = row_ptr[rp_offset + row_idx + 1];

        // Accumulate all the points for that bucket
        Jacobian sum = inf;

        for (uint k = row_begin; k < row_end; k++) {
            const uint val_idx_offset = (subtask_idx + subtask_offset) * input_size + k;
            const uint idx = val_idx[val_idx_offset];

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
