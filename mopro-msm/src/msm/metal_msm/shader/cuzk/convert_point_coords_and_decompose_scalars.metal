#include <metal_stdlib>
#include <metal_math>
#include "barrett_reduction.metal"
#include "extract_word_from_bytes_le.metal"
#include "../misc/get_constant.metal"
using namespace metal;

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
    #include <metal_logging>
    constant os_log logger_kernel(/*subsystem=*/"pt_conversion", /*category=*/"metal");
    #define LOG_DEBUG(...) logger_kernel.log_debug(__VA_ARGS__)
#else
    #define LOG_DEBUG(...) ((void)0)
#endif

kernel void convert_point_coords_and_decompose_scalars(
    device const uint* coords           [[buffer(0)]],
    device const uint* scalars          [[buffer(1)]],
    constant uint& input_size           [[buffer(2)]],
    device BigInt* point_x              [[buffer(3)]],
    device BigInt* point_y              [[buffer(4)]],
    device uint* chunks                 [[buffer(5)]],
    device const uint* num_y_workgroups [[buffer(6)]],
    uint3 gid                           [[thread_position_in_grid]]
) {
    uint gidx = gid.x;
    uint gidy = gid.y;
    uint id   = gidx * (*num_y_workgroups) + gidy;

    // 1) Convert coords to BigInt in Montgomery form
    // We read 16 halfwords for x and 16 halfwords for y. 
    uint x_bytes[16];
    uint y_bytes[16];

    // offset within coords array for this point:
    //  coords layout: [x0_32, x1_32, ..., x7_32, y0_32, y1_32, ..., y7_32]
    //  so each point uses 16 x 32-bit = 16 indices.
    uint base_offset = id * 16u;

    for (uint i = 0u; i < 8u; i++) {
        // coords[base_offset + i] is the i-th 32-bit chunk of x
        // coords[base_offset + 8 + i] is the i-th 32-bit chunk of y
        uint x_val = coords[base_offset + i];
        uint y_val = coords[base_offset + 8u + i];

        x_bytes[15 - (i * 2)]     = x_val & 0xFFFFu;
        x_bytes[15 - (i * 2) - 1] = x_val >> 16u;

        y_bytes[15 - (i * 2)]     = y_val & 0xFFFFu;
        y_bytes[15 - (i * 2) - 1] = y_val >> 16u;
    }

    BigInt x_bigint = bigint_zero();
    BigInt y_bigint = bigint_zero();

    for (uint i = 0; i < NUM_LIMBS - 1u; i++) {
        x_bigint.limbs[i] = extract_word_from_bytes_le(x_bytes, i, LOG_LIMB_SIZE);
        y_bigint.limbs[i] = extract_word_from_bytes_le(y_bytes, i, LOG_LIMB_SIZE);
    }

    uint shift = (((NUM_LIMBS * LOG_LIMB_SIZE) - 256u) + 16u) - LOG_LIMB_SIZE;

    x_bigint.limbs[NUM_LIMBS - 1] = x_bytes[0] >> shift;
    y_bigint.limbs[NUM_LIMBS - 1] = y_bytes[0] >> shift;

    // Convert x,y to Montgomery form: X = x * R mod p, Y = y * R mod p.
    BigIntWide r = get_r();
    BigInt x_mont = field_mul(bigint_to_wide(x_bigint), r);
    BigInt y_mont = field_mul(bigint_to_wide(y_bigint), r);

    // Store them in point_x, point_y
    point_x[id] = x_mont;
    point_y[id] = y_mont;

    // 2) Decompose scalars: read 8 32-bit values => 16 halfwords in `scalar_bytes`.
    uint scalar_bytes[16];
    for (uint i = 0; i < 8u; i++) {
        uint s = scalars[id * 8u + i];
        uint hi = s >> 16u;
        uint lo = s & 0xFFFFu;
        scalar_bytes[15u - (i * 2u)] = lo;
        scalar_bytes[15u - (i * 2u) - 1u] = hi;
    }

    // Extract wNAF representation. each chunk is CHUNK_SIZE bits from the scalar.
    uint chunks_arr[NUM_SUBTASKS];
    for (uint i = 0; i < NUM_SUBTASKS - 1u; i++) {
        chunks_arr[i] = extract_word_from_bytes_le(scalar_bytes, i, CHUNK_SIZE);
    }

    // The last chunk is special if (NUM_SUBTASKS * CHUNK_SIZE > 256)
    chunks_arr[NUM_SUBTASKS - 1] =
        scalar_bytes[0] >> (((NUM_SUBTASKS * CHUNK_SIZE - 256u) + 16u) - CHUNK_SIZE);

    // 3) Signed wNAF in the range [−(l−1), (l−1)]
    uint l = NUM_COLUMNS;
    uint s = l / 2u;

    int signed_slices[NUM_SUBTASKS];
    uint carry = 0;
    for (uint i = 0; i < NUM_SUBTASKS; i++) {
        int slice_val = int(chunks_arr[i] + carry);
        if (slice_val >= int(s)) {
            slice_val = (int(l) - slice_val) * (-1);
            carry = 1u;
        }
        else {
            carry = 0u;
        }
        signed_slices[i] = slice_val;
    }

    // Store final values into chunks
    for (uint i = 0; i < NUM_SUBTASKS; i++) {
        uint offset = i * input_size;
        // shift negative slices by +s to keep them in [0, l) range
        chunks[id + offset] = uint(signed_slices[i]) + s;
    }
}
