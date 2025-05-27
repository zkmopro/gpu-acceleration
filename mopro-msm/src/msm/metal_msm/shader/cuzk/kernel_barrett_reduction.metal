#include <metal_stdlib>
#include <metal_math>
#include "barrett_reduction.metal"
using namespace metal;

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
    #include <metal_logging>
    constant os_log logger_kernel(/*subsystem=*/"ketnel_barret_reduction", /*category=*/"metal");
    #define LOG_DEBUG(...) logger_kernel.log_debug(__VA_ARGS__)
#else
    #define LOG_DEBUG(...) ((void)0)
#endif

kernel void run(
    device BigIntExtraWide* a   [[ buffer(0) ]],
    device BigInt* res          [[ buffer(1) ]],
    uint gid                    [[ thread_position_in_grid ]]
) {
    *res = barrett_reduce(*a);
    LOG_DEBUG("pointer: %p", res);
}
