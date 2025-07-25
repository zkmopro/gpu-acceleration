#include "barrett_reduction.metal"
#include <metal_math>
#include <metal_stdlib>
using namespace metal;

kernel void test_field_mul(
    device BigIntWide* a [[buffer(0)]],
    device BigIntWide* b [[buffer(1)]],
    device BigInt* res [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    *res = field_mul(*a, *b);
}
