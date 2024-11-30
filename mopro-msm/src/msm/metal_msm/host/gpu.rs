use metal::*;

pub fn get_default_device() -> metal::Device {
    Device::system_default().expect("No device found")
}

pub fn create_buffer(device: &Device, data: &Vec<u32>) -> metal::Buffer {
    device.new_buffer_with_data(
        unsafe { std::mem::transmute(data.as_ptr()) },
        (data.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::CPUCacheModeDefaultCache,
    )
}

pub fn read_buffer(result_buf: &metal::Buffer, num_u32s: usize) -> Vec<u32> {
    let ptr = result_buf.contents() as *const u32;
    let result_limbs: Vec<u32>;

    // Check if ptr is not null
    if !ptr.is_null() {
        result_limbs = unsafe { std::slice::from_raw_parts(ptr, num_u32s) }.to_vec();
    } else {
        panic!("Pointer is null");
    }
    result_limbs
}

pub fn create_empty_buffer(device: &Device, size: usize) -> metal::Buffer {
    let data = vec![0u32; size];
    create_buffer(device, &data)
}

// From metal-rs
pub fn create_counter_sample_buffer(device: &Device, num_samples: usize) -> CounterSampleBuffer {
    let counter_sample_buffer_desc = metal::CounterSampleBufferDescriptor::new();
    counter_sample_buffer_desc.set_storage_mode(metal::MTLStorageMode::Shared);
    counter_sample_buffer_desc.set_sample_count(num_samples as u64);
    let counter_sets = device.counter_sets();

    let timestamp_counter = counter_sets.iter().find(|cs| cs.name() == "timestamp");

    counter_sample_buffer_desc
        .set_counter_set(timestamp_counter.expect("No timestamp counter found"));

    device
        .new_counter_sample_buffer_with_descriptor(&counter_sample_buffer_desc)
        .unwrap()
}

pub fn handle_compute_pass_sample_buffer_attachment(
    compute_pass_descriptor: &ComputePassDescriptorRef,
    counter_sample_buffer: &CounterSampleBufferRef,
) {
    let sample_buffer_attachment_descriptor = compute_pass_descriptor
        .sample_buffer_attachments()
        .object_at(0)
        .unwrap();

    sample_buffer_attachment_descriptor.set_sample_buffer(counter_sample_buffer);
    sample_buffer_attachment_descriptor.set_start_of_encoder_sample_index(0);
    sample_buffer_attachment_descriptor.set_end_of_encoder_sample_index(1);
}

pub fn resolve_samples_into_buffer(
    command_buffer: &CommandBufferRef,
    counter_sample_buffer: &CounterSampleBufferRef,
    destination_buffer: &BufferRef,
    num_samples: usize,
) {
    let blit_encoder = command_buffer.new_blit_command_encoder();
    blit_encoder.resolve_counters(
        counter_sample_buffer,
        metal::NSRange::new(0_u64, num_samples as u64),
        destination_buffer,
        0_u64,
    );
    blit_encoder.end_encoding();
}

// From metal-rs
pub fn handle_timestamps(
    resolved_sample_buffer: &BufferRef,
    cpu_start: u64,
    cpu_end: u64,
    gpu_start: u64,
    gpu_end: u64,
    num_samples: usize,
) {
    let samples = unsafe {
        std::slice::from_raw_parts(resolved_sample_buffer.contents() as *const u64, num_samples)
    };
    let pass_start = samples[0];
    let pass_end = samples[1];
    println!("samples: {:?}", samples);

    let cpu_time_span = cpu_end - cpu_start;
    let gpu_time_span = gpu_end - gpu_start;

    let micros = microseconds_between_begin(pass_start, pass_end, gpu_time_span, cpu_time_span);
    println!("Compute pass duration: {} µs", micros);
}

// From metal-rs
/// <https://developer.apple.com/documentation/metal/gpu_counters_and_counter_sample_buffers/converting_gpu_timestamps_into_cpu_time>
pub fn microseconds_between_begin(
    begin: u64,
    end: u64,
    gpu_time_span: u64,
    cpu_time_span: u64,
) -> f64 {
    let time_span = (end as f64) - (begin as f64);
    let nanoseconds = time_span / (gpu_time_span as f64) * (cpu_time_span as f64);
    nanoseconds / 1000.0
}
