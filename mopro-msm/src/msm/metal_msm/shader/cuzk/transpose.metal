#include <metal_stdlib>
using namespace metal;

kernel void transpose(
    device const uint* all_csr_col_idx       [[buffer(0)]],
    device atomic_uint* all_csc_col_ptr      [[buffer(1)]],
    device uint* all_csc_val_idxs            [[buffer(2)]],
    device atomic_uint* all_curr             [[buffer(3)]],
    constant uint2& params                   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // Subtask index (equivalent to thread ID)
    const uint subtask_idx = gid;
    
    const uint n = params.x;    // Number of columns
    const uint input_size = params.y;   // Input size
    
    // Calculate buffer offsets for this subtask
    const uint ccp_offset = subtask_idx * (n + 1u);
    const uint cci_offset = subtask_idx * input_size;
    const uint curr_offset = subtask_idx * n;
    
    // Phase 1: Count non-zero elements in each column
    for (uint j = 0; j < input_size; j++) {
        const uint col = all_csr_col_idx[cci_offset + j];
        atomic_fetch_add_explicit(
            &all_csc_col_ptr[ccp_offset + col + 1u], 
            1u, 
            memory_order_relaxed
        );
    }
    
    // Phase 2: Prefix sum for column pointers
    for (uint i = 1; i < n + 1u; i++) {
        const uint prev_count = atomic_load_explicit(
            &all_csc_col_ptr[ccp_offset + i - 1u], 
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &all_csc_col_ptr[ccp_offset + i], 
            prev_count, 
            memory_order_relaxed
        );
    }
    
    // Phase 3: Rearrange elements into CSC format
    uint val = 0u;
    for (uint j = 0; j < input_size; j++) {
        const uint col = all_csr_col_idx[cci_offset + j];
        
        // Get current position for this column
        const uint loc = atomic_load_explicit(
            &all_csc_col_ptr[ccp_offset + col], 
            memory_order_relaxed
        );
        
        // Calculate final position and update current counter
        const uint curr_pos = atomic_fetch_add_explicit(
            (device atomic_uint*)&all_curr[curr_offset + col],  // Cast to atomic_uint pointer
            1u, 
            memory_order_relaxed
        );
        
        // Store the value index in CSC format
        all_csc_val_idxs[cci_offset + loc + curr_pos] = val;
        val++;
    }
}
