// Scalar decomposition into signed window digits.
//
// Point coordinates need no GPU-side processing at all: the host passes
// arkworks' internal Montgomery representation (R = 2^256, identical to the
// GPU Montgomery radix) straight through in the coords buffer, which SMVP
// reads directly as interleaved affine points.
#include "../misc/get_constant.metal"
#include "extract_word_from_bytes_le.metal"
#include <metal_math>
#include <metal_stdlib>
using namespace metal;

kernel void decompose_scalars(
    device const uint* scalars [[buffer(0), access(read)]],
    device uint* chunks [[buffer(1), access(write)]],
    constant uint4& params [[buffer(2), access(read)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 threadgroup_size [[threadgroups_per_grid]])
{
    const uint input_size = params[0];
    const uint window_size = params[1];
    const uint num_columns = params[2];
    const uint num_subtask = params[3];

    uint gidx = gid.x;
    uint gidy = gid.y;
    uint id = gidx * threadgroup_size.y + gidy;

    // Read 8 32-bit words => 16 halfwords (big-endian halfword order) in
    // `scalar_bytes` for windowed extraction.
    uint scalar_bytes[16];

#pragma unroll(8)
    for (uint i = 0; i < 8u; i++) {
        uint s = scalars[id * 8u + i];
        uint hi = s >> 16u;
        uint lo = s & 0xFFFFu;
        scalar_bytes[15u - (i * 2u)] = lo;
        scalar_bytes[15u - (i * 2u) - 1u] = hi;
    }

    // Extract wNAF representation. each chunk is window_size bits from the scalar.
    uint l = num_columns;
    uint s = l / 2u;
    uint carry = 0;

    for (uint i = 0; i < num_subtask; i++) {
        // Extract chunk on-demand
        uint chunk_val;
        if (i < num_subtask - 1u) {
            chunk_val = extract_word_from_bytes_le(scalar_bytes, i, window_size);
        } else {
            // The last chunk is special
            chunk_val = scalar_bytes[0] >> (((num_subtask * window_size - 256u) + 16u) - window_size);
        }

        // Process signed wNAF directly
        int slice_val = int(chunk_val + carry);
        if (slice_val >= int(s)) {
            slice_val = (int(l) - slice_val) * (-1);
            carry = 1u;
        } else {
            carry = 0u;
        }

        // Store final value directly
        uint offset = i * input_size;
        chunks[id + offset] = uint(slice_val) + s;
    }
}
