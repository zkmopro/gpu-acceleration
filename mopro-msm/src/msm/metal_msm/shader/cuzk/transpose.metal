// Serial transpose algo adapted from Wang et al, 2016, "Parallel
// Transposition of Sparse Data Structures".
// https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf

#include <metal_stdlib>
using namespace metal;

kernel void transpose(
    device const uint* all_csr_col_idx [[buffer(0), access(read)]],
    device atomic_uint* all_csc_col_ptr [[buffer(1), access(read_write)]],
    device uint* all_csc_val_idxs [[buffer(2), access(read_write)]],
    device uint* all_curr [[buffer(3), access(read_write)]],
    constant uint2& params [[buffer(4), access(read)]],
    uint gid [[thread_position_in_grid]])
{
    const uint subtask_idx = gid;

    const uint n = params.x; // Number of columns
    const uint input_size = params.y; // Input size

    // Calculate buffer offsets for this subtask
    const uint ccp_offset = subtask_idx * (n + 1u);
    const uint cci_offset = subtask_idx * input_size;
    const uint curr_offset = subtask_idx * n;

    // Phase 1: Count non-zero elements in each column
    for (uint j = 0u; j < input_size; j++) {
        atomic_fetch_add_explicit(
            &all_csc_col_ptr[ccp_offset + all_csr_col_idx[cci_offset + j] + 1u],
            1u,
            memory_order_relaxed);
    }

    // Phase 2: Prefix sum for column pointers
    for (uint i = 1u; i < n + 1u; i++) {
        const uint incremental_sum = atomic_load_explicit(
            &all_csc_col_ptr[ccp_offset + i - 1u],
            memory_order_relaxed);
        atomic_fetch_add_explicit(
            &all_csc_col_ptr[ccp_offset + i],
            incremental_sum,
            memory_order_relaxed);
    }

    // Phase 3: Rearrange elements into CSC format
    /// "Traverse the nonzero elements again and move them to their final
    /// positions determined by the column offsets in csc_col_ptr and their
    /// current relative positions in curr." (Wang et al, 2016).
    uint val = 0u;
    for (uint j = 0; j < input_size; j++) {
        const uint col = all_csr_col_idx[cci_offset + j];

        // Get current position for this column
        const uint loc = atomic_load_explicit(
                             &all_csc_col_ptr[ccp_offset + col],
                             memory_order_relaxed)
            + all_curr[curr_offset + col];

        all_curr[curr_offset + col]++;

        // Store the value index in CSC format
        all_csc_val_idxs[cci_offset + loc] = val;
        val++;
    }
}
