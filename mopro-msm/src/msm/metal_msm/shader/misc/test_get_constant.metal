using namespace metal;
#include <metal_stdlib>
#include "get_constant.metal"

kernel void test_get_mu(device BigInt* result) {
    *result = get_mu();
}

kernel void test_get_n0(device BigInt* result) {
    *result = get_n0();
}

kernel void test_get_r(device BigInt* result) {
    *result = get_r();
}

kernel void test_get_p(device BigInt* result) {
    *result = get_p();
}

kernel void test_get_p_wide(device BigIntWide* result) {
    *result = get_p_wide();
}

kernel void test_get_bn254_zero(device BigInt* result_x, device BigInt* result_y, device BigInt* result_z) {
    Jacobian result = get_bn254_zero();
    *result_x = result.x;
    *result_y = result.y;
    *result_z = result.z;
}

kernel void test_get_bn254_one(device BigInt* result_x, device BigInt* result_y, device BigInt* result_z) {
    Jacobian result = get_bn254_one();
    *result_x = result.x;
    *result_y = result.y;
    *result_z = result.z;
}

kernel void test_get_bn254_zero_mont(device BigInt* result_x, device BigInt* result_y, device BigInt* result_z) {
    Jacobian result = get_bn254_zero_mont();
    *result_x = result.x;
    *result_y = result.y;
    *result_z = result.z;
}

kernel void test_get_bn254_one_mont(device BigInt* result_x, device BigInt* result_y, device BigInt* result_z) {
    Jacobian result = get_bn254_one_mont();
    *result_x = result.x;
    *result_y = result.y;
    *result_z = result.z;
}
