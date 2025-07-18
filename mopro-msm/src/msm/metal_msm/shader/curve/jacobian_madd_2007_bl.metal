// source: https://github.com/geometryxyz/msl-secp256k1

using namespace metal;
#include "jacobian.metal"
#include <metal_math>
#include <metal_stdlib>

kernel void test_jacobian_madd_2007_bl(
    device BigInt* a_xr [[buffer(0)]],
    device BigInt* a_yr [[buffer(1)]],
    device BigInt* a_zr [[buffer(2)]],
    device BigInt* b_xr [[buffer(3)]],
    device BigInt* b_yr [[buffer(4)]],
    device BigInt* result_xr [[buffer(5)]],
    device BigInt* result_yr [[buffer(6)]],
    device BigInt* result_zr [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    BigInt x1 = *a_xr;
    BigInt y1 = *a_yr;
    BigInt z1 = *a_zr;
    BigInt x2 = *b_xr;
    BigInt y2 = *b_yr;

    Jacobian a;
    a.x = x1;
    a.y = y1;
    a.z = z1;
    Affine b;
    b.x = x2;
    b.y = y2;

    Jacobian res = jacobian_madd_2007_bl(a, b);
    *result_xr = res.x;
    *result_yr = res.y;
    *result_zr = res.z;
}
