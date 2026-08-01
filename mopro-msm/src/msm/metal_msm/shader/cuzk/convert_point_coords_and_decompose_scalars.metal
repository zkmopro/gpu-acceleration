#include "../mont_backend/mont.metal"
#include "extract_word_from_bytes_le.metal"
#include <metal_math>
#include <metal_stdlib>
using namespace metal;

kernel void convert_point_coords_and_decompose_scalars(
    device const uint* coords [[buffer(0), access(read)]],
    device const uint* scalars [[buffer(1), access(read)]],
    device BigInt* point_x [[buffer(2), access(write)]],
    device BigInt* point_y [[buffer(3), access(write)]],
    device uint* chunks [[buffer(4), access(write)]],
    constant uint4& params [[buffer(5), access(read)]],
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

    // 1) Load coordinates. The coords buffer holds the little-endian 32-bit
    //    words of x then y for each point, which is exactly our limb layout.
    //    coords layout: [x0_32, ..., x7_32, y0_32, ..., y7_32] per point.
    const uint base_offset = id * (2u * NUM_LIMBS);

    BigInt x_bigint;
    BigInt y_bigint;

#pragma unroll(8)
    for (uint i = 0; i < NUM_LIMBS; i++) {
        x_bigint.limbs[i] = coords[base_offset + i];
        y_bigint.limbs[i] = coords[base_offset + NUM_LIMBS + i];
    }

    // Convert x,y to Montgomery form: mont_mul(a, R^2) = a * R mod p.
    BigInt r2 = get_r2();
    point_x[id] = mont_mul_cios(x_bigint, r2);
    point_y[id] = mont_mul_cios(y_bigint, r2);

    // 2) Decompose scalars: read 8 32-bit words => 16 halfwords (big-endian
    //    halfword order) in `scalar_bytes` for windowed extraction.
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
