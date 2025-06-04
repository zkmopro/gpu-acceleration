use crate::msm::metal_msm::host::gpu::{
    create_buffer, create_empty_buffer, get_default_device, read_buffer,
};
// no runtime shader compilation, drop constants generation here
use crate::msm::metal_msm::utils::barrett_params::calc_barrett_mu;
use crate::msm::metal_msm::utils::mont_params::{calc_mont_radix, calc_nsafe, calc_rinv_and_n0};

use ark_bn254::Fq as BaseField;
use ark_ff::PrimeField;
use metal::*;
use num_bigint::BigUint;

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;
// Embed the precompiled Metal library
include!(concat!(env!("OUT_DIR"), "/built_shaders.rs"));

/// Cache of precomputed constants
static CONSTANTS_CACHE: Lazy<Mutex<HashMap<(usize, u32), MSMConstants>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Struct for Metal config
#[derive(Clone)]
pub struct MetalConfig {
    pub log_limb_size: u32,
    pub num_limbs: usize,
    pub shader_file: String,
    pub kernel_name: String,
}

/// Struct for MSM constants
#[derive(Clone)]
pub struct MSMConstants {
    pub p: BigUint,
    pub r: BigUint,
    pub rinv: BigUint,
    pub n0: u32,
    pub nsafe: usize,
    pub mu: BigUint,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            log_limb_size: 16,
            num_limbs: 16,
            shader_file: "".to_string(),
            kernel_name: "".to_string(),
        }
    }
}

/// Enhanced Metal helper that works with pre-compiled shaders
pub struct MetalHelper {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub buffers: Vec<Buffer>,
}

impl MetalHelper {
    /// Create a new Metal helper with specific device
    pub fn new() -> Self {
        let device = get_default_device();
        let command_queue = device.new_command_queue();

        Self {
            device,
            command_queue,
            buffers: Vec::new(),
        }
    }

    /// Create a new Metal helper with custom device
    pub fn with_device(device: Device) -> Self {
        let command_queue = device.new_command_queue();

        Self {
            device,
            command_queue,
            buffers: Vec::new(),
        }
    }

    /// Create a buffer in Vec<u32> and track it
    pub fn create_buffer(&mut self, data: &Vec<u32>) -> Buffer {
        let buffer = create_buffer(&self.device, data);
        self.buffers.push(buffer.clone());
        buffer
    }

    /// Create an empty buffer and track it
    pub fn create_empty_buffer(&mut self, size: usize) -> Buffer {
        let buffer = create_empty_buffer(&self.device, size);
        self.buffers.push(buffer.clone());
        buffer
    }

    /// Create thread group size
    pub fn create_thread_group_size(&self, width: u64, height: u64, depth: u64) -> MTLSize {
        MTLSize {
            width,
            height,
            depth,
        }
    }

    /// Execute a Metal compute shader with pre-compiled pipeline state
    pub fn execute_shader_with_pipeline(
        &self,
        pipeline_state: &ComputePipelineState,
        buffers: &[&Buffer],
        thread_group_count: &MTLSize,
        threads_per_threadgroup: &MTLSize,
    ) {
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_pass_descriptor = ComputePassDescriptor::new();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

        encoder.set_compute_pipeline_state(&pipeline_state);

        // Set buffers
        for (i, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buffer), 0);
        }

        encoder.dispatch_thread_groups(*thread_group_count, *threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Execute a Metal compute shader (legacy method - kept for compatibility)
    pub fn execute_shader(
        &self,
        config: &MetalConfig,
        buffers: &[&Buffer],
        thread_group_count: &MTLSize,
        threads_per_threadgroup: &MTLSize,
    ) {
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_pass_descriptor = ComputePassDescriptor::new();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);


        // Load precompiled Metal library and create pipeline
        let library = self.device.new_library_with_data(MSM_METALLIB).unwrap();
        let kernel = library
            .get_function(config.kernel_name.as_str(), None)
            .unwrap();
        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .unwrap();

        encoder.set_compute_pipeline_state(&pipeline_state);

        // Set buffers
        for (i, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buffer), 0);
        }

        encoder.dispatch_thread_groups(*thread_group_count, *threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Read results in Vec<u32> from a buffer
    pub fn read_results(&self, buffer: &Buffer, size: usize) -> Vec<u32> {
        read_buffer(buffer, size)
    }

    /// Drop all tracked buffers
    pub fn drop_all_buffers(&mut self) {
        self.buffers.clear();
    }

    /// Get device reference
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get command queue reference
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Execute multiple compute shaders in sequence with shared command buffer
    pub fn execute_shaders_batch(&self, operations: Vec<ShaderOperation>) {
        let command_buffer = self.command_queue.new_command_buffer();

        for operation in operations {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&operation.pipeline_state);

            for (i, buffer) in operation.buffers.iter().enumerate() {
                encoder.set_buffer(i as u64, Some(buffer), 0);
            }

            encoder.dispatch_thread_groups(
                operation.thread_group_count,
                operation.threads_per_threadgroup,
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}

/// Represents a single shader operation for batch execution
pub struct ShaderOperation<'a> {
    pub pipeline_state: ComputePipelineState,
    pub buffers: Vec<&'a Buffer>,
    pub thread_group_count: MTLSize,
    pub threads_per_threadgroup: MTLSize,
}

// Calculate or retrieve cached constants
pub fn get_or_calc_constants(num_limbs: usize, log_limb_size: u32) -> MSMConstants {
    let mut cache = CONSTANTS_CACHE.lock().unwrap();
    let key = (num_limbs, log_limb_size);

    if !cache.contains_key(&key) {
        let constants = calc_constants(num_limbs, log_limb_size);
        cache.insert(key, constants.clone());
        constants
    } else {
        cache.get(&key).unwrap().clone()
    }
}

// Helper to calculate constants
fn calc_constants(num_limbs: usize, log_limb_size: u32) -> MSMConstants {
    let p: BigUint = BaseField::MODULUS.try_into().unwrap();
    let r = calc_mont_radix(num_limbs, log_limb_size);
    let (rinv, n0) = calc_rinv_and_n0(&p, &r, log_limb_size);
    let nsafe = calc_nsafe(log_limb_size);
    let mu = calc_barrett_mu(&p);
    MSMConstants {
        p,
        r,
        rinv,
        n0,
        nsafe,
        mu,
    }
}
