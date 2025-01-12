using namespace metal;
#include <metal_stdlib>
#include "get_constant.metal"


kernel void test_get_mu(device BigInt* result) {
    *result = get_mu();
}

kernel void test_get_r(device BigIntWide* result) {
    *result = get_r();
}

kernel void test_get_p(device BigInt* result) {
    *result = get_p();
}

kernel void test_get_p_wide(device BigIntWide* result) {
    *result = get_p_wide();
}
