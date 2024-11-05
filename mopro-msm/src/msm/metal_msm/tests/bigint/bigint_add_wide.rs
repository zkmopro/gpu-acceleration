// adapted from: https://github.com/geometryxyz/msl-secp256k1

use crate::msm::metal::abstraction::limbs_conversion::{FromLimbs, ToLimbs};
use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
use crate::msm::metal_msm::host::shader::{compile_metal, write_constants};
use ark_ff::{BigInt, BigInteger};
use metal::*;

#[test]
#[serial_test::serial]
pub fn test_bigint_add() {
    let log_limb_size = 13;
    let num_limbs = 20;

    // Create two large numbers that will overflow when added
    let a = BigInt::new([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
    let b = BigInt::new([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);

    let max_value = BigInt::new([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);

    let mut expected = a.clone();
    expected.add_with_carry(&b);

    assert!(expected > max_value);

    let device = get_default_device();
    let a_buf = create_buffer(&device, &a.to_limbs(num_limbs, log_limb_size));
    let b_buf = create_buffer(&device, &b.to_limbs(num_limbs, log_limb_size));
    let result_buf = create_empty_buffer(&device, num_limbs + 1);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let compute_pass_descriptor = ComputePassDescriptor::new();
    let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

    write_constants(
        "../mopro-msm/src/msm/metal_msm/shader/bigint",
        num_limbs,
        log_limb_size,
        0,
        0,
    );

    let library_path = compile_metal(
        "../mopro-msm/src/msm/metal_msm/shader/bigint",
        "bigint_add_wide.metal",
    );
    let library = device.new_library_with_file(library_path).unwrap();
    let kernel = library.get_function("run", None).unwrap();

    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&kernel));

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&a_buf), 0);
    encoder.set_buffer(1, Some(&b_buf), 0);
    encoder.set_buffer(2, Some(&result_buf), 0);

    let thread_group_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result_limbs: Vec<u32> = read_buffer(&result_buf, num_limbs + 1);
    let result = BigInt::from_limbs(&result_limbs, log_limb_size);

    assert!(result.eq(&expected));
}
