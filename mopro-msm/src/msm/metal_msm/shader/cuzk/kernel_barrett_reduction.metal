using namespace metal;
#include <metal_stdlib>
#include <metal_math>
#include "barrett_reduction.metal"

#include <metal_logging>
constant os_log logger_kernel(/*subsystem=*/"ketnel_barret_reduction", /*category=*/"xyz");

kernel void run(
    device BigIntExtraWide* a [[ buffer(0) ]],
    device BigInt* res [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    *res = barrett_reduce(*a);
    logger_kernel.log_info("pointer: %p", res);
}
