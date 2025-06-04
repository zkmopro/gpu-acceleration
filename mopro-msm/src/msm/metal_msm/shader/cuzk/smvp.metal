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
    device const uint*          row_ptr         [[ buffer(0) ]],
    device const uint*          val_idx         [[ buffer(1) ]],
    device const BigInt*        new_point_x     [[ buffer(2) ]],
    device const BigInt*        new_point_y     [[ buffer(3) ]],
    device BigInt*              bucket_x        [[ buffer(4) ]],
    device BigInt*              bucket_y        [[ buffer(5) ]],
    device BigInt*              bucket_z        [[ buffer(6) ]],
    constant uint4&             params          [[ buffer(7) ]],
    uint3                       gid             [[thread_position_in_grid]],
    uint3                       tid             [[thread_position_in_threadgroup]]
) {
    const uint input_size        = params[0];
    const uint num_y_workgroups  = params[1];
    const uint num_z_workgroups  = params[2];
    const uint subtask_offset    = params[3];

    const uint gidx = gid.x;
    const uint gidy = gid.y;
    const uint gidz = gid.z;

    const uint id = (gidx * num_y_workgroups + gidy) * num_z_workgroups + gidz;
    
    const uint num_columns = NUM_COLUMNS;
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
            const uint idx = val_idx[ (subtask_idx + subtask_offset) * input_size + k ];
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
